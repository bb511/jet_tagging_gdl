#!/usr/bin/env python3
"""Status of every Optuna sweep on this machine, and the commands to finish it.

For each study: COMPLETE trials against the n_trials its run*_search.sh block asks for, and
whether a driver is still alive to produce the rest. Prints a table, then the launch commands::

    python3 scripts/experiments/sweep_status.py            # table + commands
    python3 scripts/experiments/sweep_status.py --commands # commands only
    python3 scripts/experiments/sweep_status.py --features pt,eta,phi  # one feature set
    python3 scripts/experiments/sweep_status.py --dataset top_tagging  # the replica

The catalogues of the five grid rows sit under scripts/experiments/<dataset>/<label>/,
the label being the joined feature list, the default set's included, and those of every
other row at scripts/experiments/<dataset>/. The study databases keep the default set at
the root of logs/optuna/<dataset>/ and every other set under <label>/. The top-tagging
study names carry the prefix ``top_tagging_``.

Liveness comes from the driver log, not from the trial counts, since a dead driver leaves its
last batch marked RUNNING forever.

One study may be swept by several drivers at once, one per GPU slice or pod, each tagged into a
log of its own (see submit_search_ngt.sh). Every worker's log is read, the study counts as alive
while any one of them is, and the ``workers`` column says how many were found.
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

TAIL = 15
DATASETS = ("hls4ml", "top_tagging")
# The feature set the grid rows of each data set read when no --features is given.
DEFAULT_LABEL = {"hls4ml": "px_py_pz_e", "top_tagging": "pt_etarel_phirel"}
# A row name as the catalogue names and the experiments spell it. The hls4ml pattern
# admits no underscore, so its table leaves out jedi_linear and every other such row;
# four of the nine searched top-tagging rows carry one, so their pattern admits it.
ROW_OF = {"hls4ml": "[a-z0-9]+", "top_tagging": "[a-z0-9_]+"}
STEM_RE_OF = {
    dataset: re.compile(rf"run({row}?)_search") for dataset, row in ROW_OF.items()
}
STEM_RE = STEM_RE_OF["hls4ml"]
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
HDR = "%-11s %-38s %6s %8s %6s %7s  %-8s %s"


def _regime(block: str, dataset: str = "hls4ml") -> tuple[str, str, str, str] | None:
    """(model, nconstituents, method, scheme) of one search block."""
    exp = re.search(rf"experiment={dataset}/({ROW_OF[dataset]})_(\d+)", block)
    method = re.search(r"data\.method=(\S+)", block)
    scheme = re.search(r"data\.augmentation_scheme=(\S+)", block)
    if not (exp and method and scheme):
        return None
    return exp.group(1), exp.group(2), method.group(1), scheme.group(1)


def _label(model: str, nconst: str, method: str, scheme: str) -> str:
    """The block label submit_search.sh names its driver log after.

    The same mapping lives in make_train_catalogues.VARIANTS, which names the log directories, and
    in lib.sh::block_label, which the submitters match --only against. A change has to reach all
    three.
    """
    if method == "symm_loss":
        variant = f"symm_{scheme}"
    else:
        variant = "baseline" if scheme == "none" else f"aug_{scheme}"
    return f"{model}_nconst{nconst}_{variant}"


def study_of(regime: tuple[str, str, str, str], dataset: str = "hls4ml") -> str:
    """The study name the hparams_search config builds from a block's regime; the
    studies of a data set other than hls4ml carry its name as a prefix."""
    prefix = "" if dataset == "hls4ml" else f"{dataset}_"
    return prefix + "%s_nconst%s_%s_%s" % regime


def blocks(script: str, dataset: str = "hls4ml") -> dict[str, tuple[int, int, str]]:
    """Study -> (index, n_trials, label) for each driver block in a run*_search.sh.

    The study name is not on the command line: it is rebuilt from the regime, as the
    hparams_search config builds it.
    """
    if not os.path.exists(script):
        return {}
    out = {}
    text = Path(script).read_text()
    for i, block in enumerate(
        re.split(r"^# python3 src/train\.py", text, flags=re.M)[1:], 1
    ):
        b = block.replace("\\\n", " ").replace("#", " ")
        regime = _regime(b, dataset)
        n = re.search(r"hydra\.sweeper\.n_trials=(\d+)", b)
        if regime and n:
            out[study_of(regime, dataset)] = (i, int(n.group(1)), _label(*regime))
    return out


def _tail_of(log: str) -> str:
    """The last TAIL non-blank lines of a driver log.

    The whole tail is scanned rather than the final line: a Hydra config error ends with 'Set the
    environment variable HYDRA_FULL_ERROR=1 ...', several lines below the exception.
    """
    lines = [
        line
        for line in Path(log).read_text(errors="replace").splitlines()
        if line.strip()
    ]
    return "\n".join(lines[-TAIL:])


def classify(log: str) -> tuple[str, float | None]:
    """(state, age in minutes) of a driver, from the tail of its log."""
    if not os.path.exists(log):
        return "no-log", None
    tail = _tail_of(log)
    age = (
        datetime.now() - datetime.fromtimestamp(os.path.getmtime(log))
    ).total_seconds() / 60
    if ERR.search(tail):
        return "DEAD", age
    if FINISHED.search(tail):
        return "FINISHED", age
    return "alive", age


def worker_logs(logdir: str, idx: int, label: str) -> list[Path]:
    """The live-named driver logs of one block, one per worker sweeping its study.

    A restart moves the previous log aside as ``<name>.log.<timestamp>``, so matching on
    the ``.log`` suffix already drops those: they are the record of a death, not a
    worker.
    """
    return sorted(Path(logdir).glob(f"{idx}_{label}*.log"))


def classify_workers(logs: list[Path]) -> tuple[str, float | None]:
    """(state, age) over a study's workers: alive if any is, else the most recently
    touched."""
    if not logs:
        return "no-log", None
    return max(
        (classify(str(log)) for log in logs), key=lambda sa: (sa[0] == "alive", -sa[1])
    )


def complete_of(db: str, study: str, writable: bool = False) -> int:
    """COMPLETE trials in a study.

    A driver that crashed mid-transaction leaves a hot '<db>-journal' that a read-only
    open cannot roll back, failing with 'database is locked'; ``writable`` is for
    reading copies of such databases (copied with their -journal siblings) without
    touching the live files.
    """
    if not os.path.exists(db):
        return 0
    uri = f"file:{db}" if writable else f"file:{db}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=60)) as con:
        con.execute("pragma busy_timeout=60000")
        row = con.execute(
            "select study_id from studies where study_name=?", (study,)
        ).fetchone()
        if not row:
            return 0
        counted = "select count(*) from trials where study_id=? and state='COMPLETE'"
        return con.execute(counted, (row[0],)).fetchone()[0]


def launch_command(row: dict) -> str:
    """The submit_search.sh invocation that tops this study up to its target."""
    todo = row["target"] - row["done"]
    return (
        "bash scripts/experiments/submit_search.sh"
        f" --only {row['label']} {row['script']} hydra.sweeper.n_trials={todo}"
    )


def status_of(done: int, target: int, state: str) -> str:
    """The verdict shown in the table for one study."""
    if done >= target:
        return "DONE"
    if state == "alive":
        return "running"
    if state == "FINISHED":
        return "short"  # driver exited cleanly but below target
    return "LAUNCH"  # never started, or crashed


def driver_log_dir(path: Path) -> str:
    """The driver log directory submit_search.sh gives a catalogue, named as
    lib.sh::catalogue_stem names it."""
    prefix = "top_tagging_" if path.parent.parent.name == "top_tagging" else ""
    return f"logs/searches/{prefix}{path.parent.name}_{path.stem}"


def _rows_of(
    path: Path,
    script: str,
    model: str,
    db_root: str,
    writable: bool,
    dataset: str = "hls4ml",
) -> list[dict]:
    """One row per study of a single catalogue: its trial count and its workers'
    state."""
    db = f"{db_root}/{model}.db"
    logdir = driver_log_dir(path)
    rows = []
    for study, (idx, target, label) in blocks(script, dataset).items():
        done = complete_of(db, study, writable=writable)
        logs = worker_logs(logdir, idx, label)
        state, age = classify_workers(logs)
        status = status_of(done, target, state)
        rows.append(
            dict(
                model=model,
                study=study,
                label=label,
                script=script,
                target=target,
                done=done,
                state=state,
                age=age,
                workers=len(logs),
                status=status,
            )
        )
    return rows


def features_label(spec: str | None) -> str | None:
    """The directory one feature set owns, joined as the results table joins it."""
    return "_".join(name.strip() for name in spec.split(",")) if spec else None


def db_root_of(label: str | None, dataset: str = "hls4ml") -> str:
    """Where a feature set's study databases are, the default set's at the root."""
    root = f"logs/optuna/{dataset}"
    return root if label in (None, DEFAULT_LABEL[dataset]) else f"{root}/{label}"


def catalogues_of(repo: Path, label: str | None, dataset: str) -> list[Path]:
    """The search catalogues of a feature set's directory and, without --features, of
    the data set's root, which holds the rows that pin their columns."""
    root = repo / "scripts" / "experiments" / dataset
    found = list(root.glob(f"{label or DEFAULT_LABEL[dataset]}/run*_search.sh"))
    if label is None:
        found += root.glob("run*_search.sh")
    return sorted(found)


def collect(
    repo: Path,
    db_root: str,
    writable: bool,
    label: str | None = None,
    dataset: str = "hls4ml",
) -> list[dict]:
    """One row per search block of a feature set's catalogues."""
    rows: list[dict] = []
    for path in catalogues_of(repo, label, dataset):
        match = STEM_RE_OF[dataset].match(path.stem)
        if not match:
            print(f"skipping unrecognised file name: {path.name}", file=sys.stderr)
            continue
        script = str(path.relative_to(repo))
        rows += _rows_of(path, script, match.group(1), db_root, writable, dataset)
    return sorted(rows, key=lambda r: (r["model"], r["study"]))


def _table_line(row: dict) -> str:
    """One study's line of the table."""
    todo = max(row["target"] - row["done"], 0)
    driver = (
        row["state"] if row["age"] is None else "%s (%dm)" % (row["state"], row["age"])
    )
    counts = (row["target"], row["done"], todo or "", row["workers"] or "")
    return HDR % (row["model"], row["study"], *counts, row["status"], driver)


def print_table(rows: list[dict]) -> None:
    """Every study short of its target, one line each, then a tally by status."""
    print(HDR % tuple("model study target complete todo workers status driver".split()))
    print("-" * 112)
    for row in rows:
        if row["status"] != "DONE":
            print(_table_line(row))
    tally = Counter(row["status"] for row in rows)
    print("\n" + "  ".join("%s=%d" % kv for kv in sorted(tally.items())))
    print()


def print_commands(rows: list[dict]) -> None:
    """The submit_search.sh invocations for the studies no live driver will finish."""
    need = [r for r in rows if r["status"] in ("LAUNCH", "short")]
    if not need:
        print("# nothing to launch: every study is DONE or has a live driver")
        return
    print("# %d study(ies) to launch  (run from the repository root)" % len(need))
    for row in need:
        counts = (row["done"], row["target"])
        print("# %-11s %-38s %d/%d complete" % (row["model"], row["study"], *counts))
        print(launch_command(row))


def _parse_args() -> argparse.Namespace:
    """Command line: what to print, and which copies of the databases to read."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--commands", action="store_true", help="print only the launch commands"
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="hls4ml",
        help="report the studies of this data set (default hls4ml)",
    )
    parser.add_argument(
        "--db-root",
        help="read DBs from here instead of logs/optuna/<dataset> (use "
        "copies when a crashed driver left a hot -journal)",
    )
    parser.add_argument(
        "--features",
        help="report the grid rows' studies of this comma-separated feature set, e.g. "
        "'pt,eta,phi', instead of those of the data set's default set",
    )
    return parser.parse_args()


def main() -> int:
    """Print the sweep table, then the commands that would finish the campaign."""
    args = _parse_args()
    # Anchored on this file: another working directory would give an empty table.
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    label = features_label(args.features)
    db_root = args.db_root or db_root_of(label, args.dataset)
    writable = bool(args.db_root)
    rows = collect(repo, db_root, writable, label=label, dataset=args.dataset)
    if not args.commands:
        print_table(rows)
    print_commands(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
