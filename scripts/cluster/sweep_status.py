#!/usr/bin/env python3
"""Status of every Optuna sweep on this machine, and the commands to finish it.

For each study: COMPLETE trials against the n_trials its run*_search.sh block asks for, and
whether a driver is still alive to produce the rest. Prints a table, then the launch commands::

    python3 scripts/cluster/sweep_status.py            # table + commands
    python3 scripts/cluster/sweep_status.py --commands # commands only

Liveness comes from the driver log, not from the trial counts, since a dead driver leaves its
last batch marked RUNNING forever. The whole tail is scanned rather than the final line: a Hydra
config error ends with 'Set the environment variable HYDRA_FULL_ERROR=1 ...', several lines
below the exception.
"""

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

TAIL = 15
STEM_RE = re.compile(r"run([a-z0-9]+?)_search")
ERR = re.compile(
    r"Traceback \(most recent call last\)"
    r"|HYDRA_FULL_ERROR"
    r"|mismatched input"
    r"|LexerNoViableAltException"
    r"|optuna\.exceptions\."
    r"|^[A-Za-z_.]*(Error|Exception):"
    r"|^\S+\.py: error:",
    re.M,
)
FINISHED = re.compile(r"\[HYDRA\]\s+(Best|Value)", re.M)


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``.project-root``."""
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def _regime(block: str) -> tuple[str, str, str, str] | None:
    """(model, nconstituents, method, scheme) of one search block."""
    exp = re.search(r"experiment=hls4ml/([a-z0-9]+)_(\d+)", block)
    method = re.search(r"data\.method=(\S+)", block)
    scheme = re.search(r"data\.augmentation_scheme=(\S+)", block)
    if not (exp and method and scheme):
        return None
    return exp.group(1), exp.group(2), method.group(1), scheme.group(1)


def _label(model: str, nconst: str, method: str, scheme: str) -> str:
    """The block label submit_search.sh names its driver log after."""
    if method == "symm_loss":
        variant = f"symm_{scheme}"
    else:
        variant = "baseline" if scheme == "none" else f"aug_{scheme}"
    return f"{model}_nconst{nconst}_{variant}"


def blocks(script):
    """Study -> (index, n_trials, label) for each driver block in a run*_search.sh.

    The study name is not on the command line: it is rebuilt from the regime, as the
    hparams_search config builds it.
    """
    out = {}
    if not os.path.exists(script):
        return out
    with open(script) as fh:
        text = fh.read()
    for i, block in enumerate(re.split(r"^# python3 src/train\.py", text, flags=re.M)[1:], 1):
        b = block.replace("\\\n", " ").replace("#", " ")
        regime, n = _regime(b), re.search(r"hydra\.sweeper\.n_trials=(\d+)", b)
        if regime and n:
            out["%s_nconst%s_%s_%s" % regime] = (i, int(n.group(1)), _label(*regime))
    return out


def classify(log):
    """(state, age in minutes) of a driver, from the tail of its log."""
    if not os.path.exists(log):
        return "no-log", None
    with open(log, errors="replace") as fh:
        lines = [line for line in fh.read().splitlines() if line.strip()]
    tail = "\n".join(lines[-TAIL:])
    age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(log))).total_seconds() / 60
    if ERR.search(tail):
        return "DEAD", age
    if FINISHED.search(tail):
        return "FINISHED", age
    return "alive", age


def complete_of(db, study, writable=False):
    """COMPLETE trials in a study.

    A driver that crashed mid-transaction leaves a hot '<db>-journal' that a read-only open cannot
    roll back, failing with 'database is locked'; ``writable`` is for reading copies of such
    databases (copied with their -journal siblings) without touching the live files.
    """
    if not os.path.exists(db):
        return 0
    mode = "" if writable else "?mode=ro"
    con = sqlite3.connect(f"file:{db}{mode}", uri=True, timeout=60)
    con.execute("pragma busy_timeout=60000")
    row = con.execute("select study_id from studies where study_name=?", (study,)).fetchone()
    n = 0
    if row:
        n = con.execute(
            "select count(*) from trials where study_id=? and state='COMPLETE'", (row[0],)
        ).fetchone()[0]
    con.close()
    return n


def launch_command(row):
    """The submit_search.sh invocation that tops this study up to its target."""
    return "bash scripts/cluster/submit_search.sh --only %s %s hydra.sweeper.n_trials=%d" % (
        row["label"],
        row["script"],
        row["target"] - row["done"],
    )


def status_of(done, target, state):
    """The verdict shown in the table for one study."""
    if done >= target:
        return "DONE"
    if state == "alive":
        return "running"
    if state == "FINISHED":
        return "short"  # driver exited cleanly but below target
    return "LAUNCH"  # never started, or crashed


def collect(repo, db_root, writable):
    """One row per search block found under scripts/hls4ml/."""
    rows = []
    for path in sorted(repo.glob("scripts/hls4ml/run*_search.sh")):
        script = str(path.relative_to(repo))
        m = STEM_RE.match(path.stem)
        if not m:
            print(f"skipping unrecognised file name: {script}", file=sys.stderr)
            continue
        db = f"{db_root}/{m.group(1)}.db"
        for study, (idx, target, label) in blocks(script).items():
            done = complete_of(db, study, writable=writable)
            state, age = classify(f"logs/searches/hls4ml_{path.stem}/{idx}_{label}.log")
            rows.append(
                dict(
                    model=m.group(1),
                    study=study,
                    label=label,
                    script=script,
                    target=target,
                    done=done,
                    state=state,
                    age=age,
                    status=status_of(done, target, state),
                )
            )
    return sorted(rows, key=lambda r: (r["model"], r["study"]))


def print_table(rows):
    hdr = "%-11s %-38s %6s %8s %6s  %-8s %s"
    print(hdr % ("model", "study", "target", "complete", "todo", "status", "driver"))
    print("-" * 104)
    for r in rows:
        if r["status"] == "DONE":
            continue
        todo = max(r["target"] - r["done"], 0)
        drv = r["state"] if r["age"] is None else "%s (%dm)" % (r["state"], r["age"])
        print(hdr % (r["model"], r["study"], r["target"], r["done"], todo or "", r["status"], drv))
    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print("\n" + "  ".join("%s=%d" % kv for kv in sorted(tally.items())))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", action="store_true", help="print only the launch commands")
    ap.add_argument(
        "--db-root",
        help="read DBs from here instead of logs/optuna/hls4ml (use "
        "copies when a crashed driver left a hot -journal)",
    )
    args = ap.parse_args()

    # Anchored on .project-root: globbing from the wrong CWD would silently yield
    # an empty table instead of an error.
    repo = find_repo_root(Path(__file__).resolve().parent)
    os.chdir(repo)

    rows = collect(repo, args.db_root or "logs/optuna/hls4ml", writable=bool(args.db_root))
    if not args.commands:
        print_table(rows)

    need = [r for r in rows if r["status"] in ("LAUNCH", "short")]
    if not need:
        print("# nothing to launch - every study is DONE or has a live driver")
        return 0
    print("# %d study(ies) to launch  (run from the repository root)" % len(need))
    for r in need:
        print("# %-11s %-38s %d/%d complete" % (r["model"], r["study"], r["done"], r["target"]))
        print(launch_command(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
