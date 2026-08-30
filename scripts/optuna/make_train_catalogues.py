#!/usr/bin/env python3
"""Generate the run<model>_train.sh and run<model>_eval.sh catalogues.

Every architecture is trained on a 3 x 7 grid: three constituent counts
(16, 50, 150) crossed with seven regimes (the unaugmented baseline, SO(2),
SO(3) and Lorentz augmentation, and the same three groups as a
symmetry-consistency loss). This writes the 21 training commands of that grid
and the 21 evaluation commands that read their checkpoint manifests.

The hyperparameters come from the Optuna studies under ``logs/optuna/hls4ml/``,
one study per (model, constituent count, regime), named
``<model>_nconst<N>_<method>_<scheme>``. A block whose study is missing or
still empty carries no ``algorithm.*`` overrides at all: it trains the
experiment config's own values, which is the transferred-hyperparameter
ablation, and its banner says so. ``--transferred`` forces that state for every
block and writes to ``run<model>_{train,eval}_transferred.sh``, so that the
ablation can be run alongside the tuned catalogues rather than over them.

The generated text is the record of what was run, so regenerate rather than
edit by hand; ``--dry-run`` reports which files a regeneration would change
without touching any of them.

Usage (from anywhere; paths are anchored to the repository root)::

    python3 scripts/optuna/make_train_catalogues.py
    python3 scripts/optuna/make_train_catalogues.py --dry-run
    python3 scripts/optuna/make_train_catalogues.py --transferred
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

MODELS = ["deepsets", "mlp", "jetformer", "lorentznet"]
NCONSTITUENTS = [16, 50, 150]

# (label, data.method, data.augmentation_scheme), in the order every catalogue,
# every log directory and every results table uses.
VARIANTS = [
    ("baseline", "augmentation", "none"),
    ("aug_so2", "augmentation", "so2"),
    ("aug_so3", "augmentation", "so3"),
    ("aug_lorentz", "augmentation", "lorentz"),
    ("symm_so2", "symm_loss", "so2"),
    ("symm_so3", "symm_loss", "so3"),
    ("symm_lorentz", "symm_loss", "lorentz"),
]

# Folds per training run. The searches use one fold; the catalogues use five,
# which is what the paired bootstrap in scripts/analysis/add_pvalues.py needs.
N_FOLDS = 5

BANNER = "# " + 72 * "="
SEP = "# " + 72 * "-"


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def study_name(model: str, nconst: int, method: str, scheme: str) -> str:
    """The study name configs/hparams_search/hls4ml/<model>_optuna.yaml builds."""
    return f"{model}_nconst{nconst}_{method}_{scheme}"


def quote(value: str) -> str:
    """Single-quote a value the shell would otherwise treat as a glob or a brace."""
    return f"'{value}'" if any(c in value for c in "[]{},") else value


def best_trial(db: Path, study: str) -> tuple[int, float, dict[str, str]] | None:
    """Best trial of ``study``, as (number, objective value, params), or None.

    Optuna is imported here rather than at module scope so that the generator runs with no study
    databases present, which is the state the checked-in catalogues were generated in.
    """
    if not db.exists():
        return None
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        loaded = optuna.load_study(study_name=study, storage=f"sqlite:///{db}")
        trial = loaded.best_trial
    except (KeyError, ValueError):  # no such study, or no completed trial in it
        return None
    return trial.number, float(trial.value), {k: str(v) for k, v in trial.params.items()}


def overrides_of(params: dict[str, str]) -> list[str]:
    """The search-space parameters as hydra override tokens, sorted by key.

    The Optuna parameter names are already the override paths (the sweeper
    takes them straight from ``hydra.sweeper.params``), so there is nothing to
    remap: only the list values need quoting.
    """
    return [f"{k}={quote(params[k])}" for k in sorted(params)]


def run_dir(task: str, model: str, nconst: int, label: str, suffix: str) -> str:
    return f"logs/{task}/hls4ml/{model}/nconst{nconst}/{label}{suffix}"


def _block(header: list[str], entry: str, body: list[str]) -> list[str]:
    """One commented block: banner, header lines, then the wrapped command."""
    lines = [SEP, *header, SEP, f"# python3 {entry} \\"]
    lines += [f"#     {tok} \\" for tok in body[:-1]] + [f"#     {body[-1]}", ""]
    return lines


def _provenance(trial: tuple[int, float, dict[str, str]] | None, study: str) -> str:
    if trial is None:
        return "# transferred from experiment config (no study yet)"
    return f"# best trial {trial[0]} of {study}, val/acc_best={trial[1]:.6f}"


def train_block(
    model: str,
    nconst: int,
    variant: tuple[str, str, str],
    trial: tuple[int, float, dict[str, str]] | None,
    suffix: str,
) -> list[str]:
    """One training command, folds included, writing to its own hydra run dir."""
    label, method, scheme = variant
    body = [
        f"experiment=hls4ml/{model}_{nconst}",
        f"data.method={method}",
        f"data.augmentation_scheme={scheme}",
        f"data.n_folds={N_FOLDS}",
    ]
    body += overrides_of(trial[2]) if trial else []
    body += [
        "trainer=gpu",
        "trainer.devices=[0]",
        f"hydra.run.dir={run_dir('train', model, nconst, label, suffix)}",
    ]
    header = [
        f"# {model}  nconst={nconst}  {label}  (method={method}, group={scheme})",
        _provenance(trial, study_name(model, nconst, method, scheme)),
    ]
    return _block(header, "src/train.py", body)


def eval_block(
    model: str,
    nconst: int,
    variant: tuple[str, str, str],
    trial: tuple[int, float, dict[str, str]] | None,
    suffix: str,
) -> list[str]:
    """One evaluation command, reading the manifest its training block wrote.

    It repeats the training block's ``algorithm.*`` overrides because the model is rebuilt from the
    config before the checkpoint is loaded into it.
    """
    label, method, scheme = variant
    manifest = f"{run_dir('train', model, nconst, label, suffix)}/checkpoint_manifest.json"
    body = [
        f"experiment=hls4ml/{model}_{nconst}",
        f"data.method={method}",
        f"data.augmentation_scheme={scheme}",
        f"data.n_folds={N_FOLDS}",
    ]
    body += overrides_of(trial[2]) if trial else []
    body += [
        f"checkpoint_manifest_path={manifest}",
        "callbacks.results_table.invariance_metric=wasserstein",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"hydra.run.dir={run_dir('eval', model, nconst, label, suffix)}",
    ]
    header = [
        f"# {model}  nconst={nconst}  {label}  (method={method}, group={scheme})",
        _provenance(trial, study_name(model, nconst, method, scheme)),
    ]
    return _block(header, "src/eval.py", body)


TRANSFERRED_NOTE = [
    "# No block carries hyperparameter overrides: this is the transferred-",
    "# hyperparameter ablation, run at the experiment configs' own values.",
]
TUNED_NOTE = [
    "# Each block carries the best trial of its own study. Blocks whose study is",
    "# missing carry no overrides and say so: they are the transferred ablation.",
]


def file_header(model: str, task: str, transferred: bool) -> list[str]:
    what = "TRAINING" if task == "train" else "EVALUATION"
    return [
        "#!/usr/bin/env bash",
        BANNER,
        f"# {model.upper()} {what} COMMANDS",
        BANNER,
        f"# One block per (constituent count, regime): 3 x 7 = 21 {task} runs.",
        "#",
        "# Generated from the study databases under logs/optuna/hls4ml/ by",
        "# scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.",
        *(TRANSFERRED_NOTE if transferred else TUNED_NOTE),
        "#",
        "# Every command is commented out. Copy one to run it by hand, or feed the",
        f"# whole file to scripts/cluster/submit_{task}.sh, which parses these blocks",
        "# and can queue them on a GPU or send them to slurm.",
        "",
    ]


def render(model: str, task: str, dbs: Path, transferred: bool) -> str:
    """The full text of one catalogue."""
    suffix = "_transferred" if transferred else ""
    make_block = train_block if task == "train" else eval_block
    out = file_header(model, task, transferred)
    for nconst in NCONSTITUENTS:
        out += [BANNER, f"# {model}  nconst={nconst}", BANNER, ""]
        for variant in VARIANTS:
            study = study_name(model, nconst, variant[1], variant[2])
            trial = None if transferred else best_trial(dbs / f"{model}.db", study)
            out += make_block(model, nconst, variant, trial, suffix)
    return "\n".join(out).rstrip("\n") + "\n"


def count_commands(lib: Path, text: str) -> int:
    """Commands lib.sh's own parser finds in ``text``.

    The submitters and this generator have to agree on what a block is, so the generated text is
    fed back through the parser they use rather than through a second implementation of it here.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".sh") as fh:
        fh.write(text)
        fh.flush()
        script = f'. "{lib}"; extract_cmds "$1"'
        found = subprocess.run(
            ["bash", "-c", script, "_", fh.name], capture_output=True, text=True, check=True
        )
    return len([line for line in found.stdout.splitlines() if line.strip()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dbs",
        default=None,
        help="Study databases directory (default <repo>/logs/optuna/hls4ml).",
    )
    parser.add_argument(
        "--out", default=None, help="Where the catalogues go (default <repo>/scripts/hls4ml)."
    )
    parser.add_argument(
        "--transferred",
        action="store_true",
        help="Ignore the studies and write the *_transferred.sh ablation.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report which files would change; write nothing."
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    dbs = Path(args.dbs).expanduser() if args.dbs else repo / "logs" / "optuna" / "hls4ml"
    out = Path(args.out).expanduser() if args.out else repo / "scripts" / "hls4ml"
    lib = repo / "scripts" / "cluster" / "lib.sh"

    def show(p: Path) -> str:
        """Repo-relative when it is inside the repo, absolute otherwise."""
        try:
            return str(p.relative_to(repo))
        except ValueError:
            return str(p)

    expected = len(NCONSTITUENTS) * len(VARIANTS)
    stale = 0
    for model in MODELS:
        for task in ("train", "eval"):
            text = render(model, task, dbs, args.transferred)
            found = count_commands(lib, text)
            if found != expected:
                print(
                    f"error: {model} {task}: lib.sh parsed {found} commands, expected "
                    f"{expected}",
                    file=sys.stderr,
                )
                return 1
            suffix = "_transferred" if args.transferred else ""
            path = out / f"run{model}_{task}{suffix}.sh"
            if args.dry_run:
                same = path.exists() and path.read_text() == text
                stale += 0 if same else 1
                print(
                    f"{show(path)}: {found} commands, "
                    f"{'unchanged' if same else 'WOULD CHANGE'}"
                )
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {show(path)}: {found} commands")
    if stale:
        print(f"\n{stale} file(s) differ from a fresh generation.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
