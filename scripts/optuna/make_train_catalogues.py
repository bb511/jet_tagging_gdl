#!/usr/bin/env python3
"""Generate the run<model>_train.sh and run<model>_eval.sh catalogues.

DeepSets, MLP and JetFormer are trained on a 3 x 7 grid:
* three constituent counts (16, 50, 150)
* seven regimes:
    - the unaugmented baseline
    - SO(2), SO(3) and Lorentz augmentation
    - SO(2), SO(3) and Lorentz symmetry loss

Symmetry-embedded architectures are trained on the baseline only.
The hyperparameter optimisation is located at ``logs/optuna/hls4ml``, or under
``logs/optuna/hls4ml/<label>/`` for a feature set other than the four-momentum.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MODELS = ["deepsets", "mlp", "jetformer", "lorentznet", "pelican", "pelican_lorentz"]
NCONSTITUENTS = [16, 50, 150]

# (label, data.method, data.augmentation_scheme), in the order every catalogue,
# log directory and results table uses.
VARIANTS = [
    ("baseline", "augmentation", "none"),
    ("aug_so2", "augmentation", "so2"),
    ("aug_so3", "augmentation", "so3"),
    ("aug_lorentz", "augmentation", "lorentz"),
    ("symm_so2", "symm_loss", "so2"),
    ("symm_so3", "symm_loss", "so3"),
    ("symm_lorentz", "symm_loss", "lorentz"),
]

# LorentzNet and PELICAN (with and without the beams) embed the symmetry rather than acquiring
# it, so they are the architectures trained at baseline alone, never under augmentation or the SCL.
EMBEDDED = {"lorentznet", "pelican", "pelican_lorentz"}

VARIANTS_OF = {
    model: VARIANTS[:1] if model in EMBEDDED else VARIANTS for model in MODELS
}

# The searches use one fold; five is what the results table callback needs to put a confidence
# interval on a run's numbers.
N_FOLDS = 5

BANNER = "# " + 72 * "="
SEP = "# " + 72 * "-"

# (trial number, objective value, params) of the study a block was tuned on.
Trial = tuple[int, float, dict[str, str]]
Variant = tuple[str, str, str]


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``.project-root``."""
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


def best_trial(db: Path, study: str) -> Trial | None:
    """Best trial of ``study``, or None when there is no such study and no completed
    trial.

    Optuna is imported here, not at module scope, so the generator runs with no study
    database present, which is the state the checked-in catalogues were generated in.
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
    return (
        trial.number,
        float(trial.value),
        {k: str(v) for k, v in trial.params.items()},
    )


def overrides_of(params: dict[str, str]) -> list[str]:
    """The search-space parameters as hydra override tokens, sorted by key.

    Optuna parameter names are already the override paths (taken straight from
    ``hydra.sweeper.params``), so only the list values need quoting.
    """
    return [f"{k}={quote(params[k])}" for k in sorted(params)]


@dataclass(frozen=True)
class Layout:
    """Where one feature set's catalogues, studies and runs live.

    The four-momentum set, which every experiment config reads, sits at the root of each
    of those three trees; any other set is held apart under its own label, the
    underscore join of its feature names that ``src/utils/results_table.py`` also uses
    for ``results/<label>/``.
    """

    features: tuple[str, ...] | None = None
    transferred: bool = False

    @property
    def label(self) -> str | None:
        """The directory component of this feature set, None for the four-momentum."""
        return "_".join(self.features) if self.features else None

    @property
    def suffix(self) -> str:
        """What tells a transferred-hyperparameter file and run directory apart."""
        return "_transferred" if self.transferred else ""

    def under(self, root: str) -> str:
        """``root`` itself, or the subdirectory of it this feature set owns."""
        return f"{root}/{self.label}" if self.label else root

    def run_dir(self, task: str, model: str, nconst: int, label: str) -> str:
        """Where one block's hydra run directory goes under logs/."""
        base = self.under(f"logs/{task}/hls4ml")
        return f"{base}/{model}/nconst{nconst}/{label}{self.suffix}"

    def overrides(self) -> list[str]:
        """The feature-set override every block of this layout carries, if any."""
        if not self.features:
            return []
        return [f"data.features={quote('[' + ','.join(self.features) + ']')}"]


def _block(header: list[str], entry: str, body: list[str]) -> list[str]:
    """One commented block: banner, header lines, then the wrapped command."""
    lines = [SEP, *header, SEP, f"# python3 {entry} \\"]
    lines += [f"#     {tok} \\" for tok in body[:-1]] + [f"#     {body[-1]}", ""]
    return lines


def _provenance(trial: Trial | None, study: str) -> str:
    """The header line saying where a block's hyperparameters came from."""
    if trial is None:
        return "# transferred from experiment config (no study yet)"
    return f"# best trial {trial[0]} of {study}, val/acc_best={trial[1]:.6f}"


def _header(
    model: str, nconst: int, variant: Variant, trial: Trial | None
) -> list[str]:
    """The two comment lines above a block: its grid cell, and its hyperparameters'
    provenance."""
    label, method, scheme = variant
    return [
        f"# {model}  nconst={nconst}  {label}  (method={method}, group={scheme})",
        _provenance(trial, study_name(model, nconst, method, scheme)),
    ]


def _common_body(
    model: str, nconst: int, variant: Variant, trial: Trial | None, layout: Layout
) -> list[str]:
    """The overrides a train and an eval block open with, in that shared order.

    An eval block repeats the training block's ``algorithm.*`` overrides, and the
    feature set with them, because the model is rebuilt from the config before the
    checkpoint is loaded into it.
    """
    _, method, scheme = variant
    body = [
        f"experiment=hls4ml/{model}_{nconst}",
        f"data.method={method}",
        f"data.augmentation_scheme={scheme}",
        f"data.n_folds={N_FOLDS}",
        *layout.overrides(),
    ]
    return body + (overrides_of(trial[2]) if trial else [])


def train_block(
    model: str, nconst: int, variant: Variant, trial: Trial | None, layout: Layout
) -> list[str]:
    """One training command, folds included, writing to its own hydra run dir."""
    body = _common_body(model, nconst, variant, trial, layout) + [
        "trainer=gpu",
        "trainer.devices=[0]",
        f"hydra.run.dir={layout.run_dir('train', model, nconst, variant[0])}",
    ]
    return _block(_header(model, nconst, variant, trial), "src/train.py", body)


def eval_block(
    model: str, nconst: int, variant: Variant, trial: Trial | None, layout: Layout
) -> list[str]:
    """One evaluation command, reading the manifest its training block wrote."""
    label = variant[0]
    train_dir = layout.run_dir("train", model, nconst, label)
    body = _common_body(model, nconst, variant, trial, layout) + [
        f"checkpoint_manifest_path={train_dir}/checkpoint_manifest.json",
        "results_table.invariance_metric=wasserstein",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"hydra.run.dir={layout.run_dir('eval', model, nconst, label)}",
    ]
    return _block(_header(model, nconst, variant, trial), "src/eval.py", body)


TRANSFERRED_NOTE = [
    "# No block carries hyperparameter overrides: this is the transferred-",
    "# hyperparameter ablation, at the experiment configs' own values.",
]
BASELINE_ONLY_NOTE = [
    "# Baseline only: this architecture embeds the symmetry rather than acquiring it,",
    "# so the augmentation and SCL regimes are not run on it.",
]


def tuned_note(layout: Layout) -> list[str]:
    """Where the blocks' hyperparameters were read from, for the reader of one file."""
    return [
        "# Each block carries the best trial of its own study under"
        f" {layout.under('logs/optuna/hls4ml')}/;",
        "# a block whose study is missing carries none and says so (transferred ablation).",
    ]


def features_note(layout: Layout) -> list[str]:
    """Which feature set a catalogue is for, and the trees it keeps to; none by
    default."""
    if layout.label is None:
        return []
    return [
        f"# Feature set [{', '.join(layout.features)}]: studies from"
        f" {layout.under('logs/optuna/hls4ml')}/, runs under"
        f" logs/{{train,eval}}/hls4ml/{layout.label}/."
    ]


def file_header(model: str, task: str, layout: Layout) -> list[str]:
    """The banner a catalogue opens with, naming the grid it covers and where it came
    from."""
    what = "TRAINING" if task == "train" else "EVALUATION"
    nvariants = len(VARIANTS_OF[model])
    grid = f"{len(NCONSTITUENTS)} x {nvariants} = {len(NCONSTITUENTS) * nvariants}"
    return [
        "#!/usr/bin/env bash",
        BANNER,
        f"# {model.upper()} {what} COMMANDS",
        BANNER,
        f"# One block per (constituent count, regime): {grid} {task} runs. Generated",
        "# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.",
        *features_note(layout),
        *([] if nvariants == len(VARIANTS) else BASELINE_ONLY_NOTE),
        *(TRANSFERRED_NOTE if layout.transferred else tuned_note(layout)),
        "#",
        "# Every command is commented out: copy one to run it by hand, or feed the whole",
        f"# file to scripts/cluster/submit_{task}.sh, which queues them on a GPU or slurm.",
        "",
    ]


def render(model: str, task: str, dbs: Path, layout: Layout) -> str:
    """The full text of one catalogue."""
    make_block = train_block if task == "train" else eval_block
    out = file_header(model, task, layout)
    for nconst in NCONSTITUENTS:
        out += [BANNER, f"# {model}  nconst={nconst}", BANNER, ""]
        for variant in VARIANTS_OF[model]:
            study = study_name(model, nconst, variant[1], variant[2])
            db = dbs / f"{model}.db"
            trial = None if layout.transferred else best_trial(db, study)
            out += make_block(model, nconst, variant, trial, layout)
    return "\n".join(out).rstrip("\n") + "\n"


def count_commands(lib: Path, text: str) -> int:
    """Commands lib.sh's own parser finds in ``text``.

    Fed back through the submitters' parser rather than a second implementation of it
    here, so the two cannot disagree about what a block is.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".sh") as fh:
        fh.write(text)
        fh.flush()
        script = f'. "{lib}"; extract_cmds "$1"'
        found = subprocess.run(
            ["bash", "-c", script, "_", fh.name],
            capture_output=True,
            text=True,
            check=True,
        )
    return len([line for line in found.stdout.splitlines() if line.strip()])


def _catalogue(
    model: str, task: str, dbs: Path, layout: Layout, lib: Path
) -> tuple[str, int]:
    """One catalogue's text, and the number of commands lib.sh's parser finds in it.

    :raises SystemExit: when the parser does not find the model's whole grid, which
        means the generated layout and the parser have drifted apart.
    """
    text = render(model, task, dbs, layout)
    found = count_commands(lib, text)
    expected = len(NCONSTITUENTS) * len(VARIANTS_OF[model])
    if found != expected:
        sys.exit(
            f"error: {model} {task}: lib.sh parsed {found} commands, expected {expected}"
        )
    return text, found


def _show(path: Path, repo: Path) -> str:
    """Repo-relative when the path is inside the repository, absolute otherwise."""
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _emit(text: str, path: Path, found: int, repo: Path, dry_run: bool) -> int:
    """Write one catalogue, or report what writing it would do; 1 when it would change
    it."""
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {_show(path, repo)}: {found} commands")
        return 0
    same = path.exists() and path.read_text() == text
    print(
        f"{_show(path, repo)}: {found} commands, {'unchanged' if same else 'WOULD CHANGE'}"
    )
    return 0 if same else 1


def models_of(layout: Layout) -> list[str]:
    """The architectures a layout has catalogues for.

    The embedded architectures read the four-momenta themselves, so no feature set
    other than that one has anything to run for them.
    """
    if layout.features is None:
        return MODELS
    return [model for model in MODELS if model not in EMBEDDED]


def _generate(dbs: Path, out: Path, repo: Path, layout: Layout, dry_run: bool) -> int:
    """Write or check every catalogue; returns how many differ from a fresh
    generation."""
    lib = repo / "scripts" / "cluster" / "lib.sh"
    stale = 0
    for model in models_of(layout):
        for task in ("train", "eval"):
            text, found = _catalogue(model, task, dbs, layout, lib)
            path = out / f"run{model}_{task}{layout.suffix}.sh"
            stale += _emit(text, path, found, repo, dry_run)
    return stale


def _parse_args() -> argparse.Namespace:
    """Command line: where the studies are, where the catalogues go, and what to
    write."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dbs",
        default=None,
        help="Study databases directory (default <repo>/logs/optuna/hls4ml"
        "[/<feature label>]).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where the catalogues go (default <repo>/scripts/hls4ml"
        "[/<feature label>]).",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature set, e.g. 'pt,eta,phi'; the default is the "
        "four-momentum layout every experiment config reads.",
    )
    parser.add_argument(
        "--transferred",
        action="store_true",
        help="Ignore the studies and write the *_transferred.sh ablation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would change; write nothing.",
    )
    return parser.parse_args()


def _features(spec: str | None) -> tuple[str, ...] | None:
    """The --features list, None when it was not given."""
    return tuple(name.strip() for name in spec.split(",")) if spec else None


def _default(given: str | None, root: Path, layout: Layout) -> Path:
    """The path the command line gave, or the one this layout owns under ``root``."""
    return Path(given).expanduser() if given else Path(layout.under(str(root)))


def main() -> int:
    """Generate every catalogue, or report which of them a regeneration would change."""
    args = _parse_args()
    repo = find_repo_root(Path(__file__).resolve().parent)
    layout = Layout(features=_features(args.features), transferred=args.transferred)
    dbs = _default(args.dbs, repo / "logs" / "optuna" / "hls4ml", layout)
    out = _default(args.out, repo / "scripts" / "hls4ml", layout)
    stale = _generate(dbs, out, repo, layout, args.dry_run)
    if stale:
        print(f"\n{stale} file(s) differ from a fresh generation.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
