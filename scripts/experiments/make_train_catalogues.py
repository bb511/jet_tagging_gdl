#!/usr/bin/env python3
"""Generate the run<model>_train.sh and run<model>_eval.sh catalogues.

DeepSets, MLP, JetFormer, the Linformer and JEDI-linear are trained on a 3 x 7 grid:
* three constituent counts (16, 50, 150)
* seven regimes:
    - the unaugmented baseline
    - SO(2), SO(3) and Lorentz augmentation
    - SO(2), SO(3) and Lorentz symmetry loss

Symmetry-embedded architectures are trained on the baseline only, and so are the four
rows on the record's jet-level observables, at 150 constituents alone.
The hyperparameter optimisation is located at ``logs/optuna/hls4ml``, or under
``logs/optuna/hls4ml/<label>/`` for a feature set other than the four-momentum.

``--dataset top_tagging`` writes the catalogues of the top-tagging replica to
``scripts/experiments/top_tagging``, at 100 constituents. The five grid rows run the
seven regimes on [pt, etarel, phirel], the columns their experiments set, or on the set
``--features`` names; the 13 rows that pin their columns run the baseline alone and have
catalogues on the default set alone. The searched rows are tuned on the studies under
``logs/optuna/top_tagging``, or under ``logs/optuna/top_tagging/<label>/`` for another
feature set, and the others on the published recipes of their experiment configs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

MODELS = [
    "deepsets",
    "mlp",
    "jetformer",
    "lorentznet",
    "pelican",
    "pelican_lorentz",
    "lgatr_slim",
    "lgatr_slim_lorentz",
    "lloca_lorentz",
    "lloca_z",
    "lloca_so2",
    "lloca_so3",
    "vn",
    "jedi_linear",
    "linformer",
    "deepsets_moments",
    "jedi_linear_moments",
    "moments_only",
]
# The MLP and the gradient-boosted trees on the record's jet-level observables, all 53 and the
# 52 without the jet rapidity: rows of no architecture, outside MODELS, which
# tests/test_algorithms.py builds on constituents.
JET_LEVEL = [
    "jet_level_all53",
    "jet_level_noeta52",
    "jet_level_bdt_all53",
    "jet_level_bdt_noeta52",
]
# Fitted by scikit-learn, so their blocks select the CPU trainer and no device, which the
# submitters' --gpu and --gpus then leave unchanged.
CPU_ROWS = {"jet_level_bdt_all53", "jet_level_bdt_noeta52"}
NCONSTITUENTS = [16, 50, 150]
# The observables are computed from the whole jet, so their rows are labelled with the one
# constituent count that never truncates it.
NCONSTITUENTS_OF = {
    model: [150] if model in JET_LEVEL else NCONSTITUENTS
    for model in [*MODELS, *JET_LEVEL]
}

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

# LorentzNet, PELICAN (with and without the beams), L-GATr-slim (with and without the spurion
# tokens), LLoCa (with the Lorentz, SO(2)xSO(1,1), SO(2) and SO(3) frames) and the vector neurons
# (on the lifts of (pt, eta, phi)) embed the symmetry rather than acquiring it, so they are the
# architectures trained at baseline alone, never under augmentation or the SCL. The three moment
# rows (DeepSets and JEDI-linear with the jet moments, and the moments-only control) are pinned to
# [pt, etarel, phirel], whose columns a rotation about the beam leaves unchanged, and are filed
# with them.
EMBEDDED = {
    "lorentznet",
    "pelican",
    "pelican_lorentz",
    "lgatr_slim",
    "lgatr_slim_lorentz",
    "lloca_lorentz",
    "lloca_z",
    "lloca_so2",
    "lloca_so3",
    "vn",
    "deepsets_moments",
    "jedi_linear_moments",
    "moments_only",
}
# The rows trained at baseline alone: the embedded rows and the jet-level rows.
BASELINE_ONLY = set(EMBEDDED) | set(JET_LEVEL)
# Trained quantisation-aware from a float row's checkpoint, with no study: outside MODELS, so
# that no catalogue of theirs is generated; scripts/experiments/hls4ml/run<row>_*.sh are
# hand-written.
QUANTISED = ["vn_hgq", "jedi_linear_hgq"]

VARIANTS_OF = {
    model: VARIANTS[:1] if model in BASELINE_ONLY else VARIANTS
    for model in [*MODELS, *JET_LEVEL]
}

# The searches use one fold; five is what the results table callback needs to put a confidence
# interval on a run's numbers.
N_FOLDS = 5

# Regimes compare fairly only on an equal trial budget, which multi-driver campaigns overshoot.
TRIAL_BUDGET = 200

DATASETS = ("hls4ml", "top_tagging")
# The feature set the grid rows of each data set read by default: the four-momentum of
# configs/data/hls4ml.yaml, and the columns the top-tagging grid experiments set.
DEFAULT_LABEL = {"hls4ml": "px_py_pz_e", "top_tagging": "pt_etarel_phirel"}
# The rows of the top-tagging replica that follow a published recipe, which no study tunes.
TOP_TAGGING_RECIPES = [
    "lorentznet",
    "pelican",
    "pelican_lorentz",
    "lgatr_slim",
    "lgatr_slim_lorentz",
    "lloca_lorentz",
    "lloca_z",
    "lloca_so2",
    "lloca_so3",
]
# Every row of the replica, at one constituent count: the recipe rows, the vector
# neurons, the five grid rows, which alone run the seven regimes and the three feature
# sets, and the moment rows.
TOP_TAGGING = [
    *TOP_TAGGING_RECIPES,
    "vn",
    "deepsets",
    "mlp",
    "jetformer",
    "jedi_linear",
    "linformer",
    "deepsets_moments",
    "jedi_linear_moments",
    "moments_only",
]
TOP_TAGGING_NCONSTITUENTS = [100]
# The trials of each top-tagging search, in a space narrowed around the row's hls4ml
# optimum.
TOP_TAGGING_BUDGET = 100

BANNER = "# " + 72 * "="
SEP = "# " + 72 * "-"

# (trial number, objective value, params) of the study a block was tuned on.
Trial = tuple[int, float, dict[str, str]]
Variant = tuple[str, str, str]


def study_name(
    model: str, nconst: int, method: str, scheme: str, dataset: str = "hls4ml"
) -> str:
    """The study name configs/hparams_search/<dataset>/<model>_optuna.yaml builds; the
    studies of a data set other than hls4ml carry its name as a prefix."""
    prefix = "" if dataset == "hls4ml" else f"{dataset}_"
    return f"{prefix}{model}_nconst{nconst}_{method}_{scheme}"


def quote(value: str) -> str:
    """Single-quote a value the shell would otherwise treat as a glob or a brace."""
    return f"'{value}'" if any(c in value for c in "[]{},") else value


def _best_within_budget(complete: list[FrozenTrial], budget: int) -> Trial | None:
    """The best of the first ``budget`` trials by trial number, the earliest on a tie;
    every study maximises val/acc_best."""
    window = sorted(complete, key=lambda t: t.number)[:budget]
    if not window:
        return None
    trial = max(window, key=lambda t: (t.value, -t.number))
    return (
        trial.number,
        float(trial.value),
        {k: str(v) for k, v in trial.params.items()},
    )


def _complete_trials(db: Path, study: str) -> list[FrozenTrial]:
    """The COMPLETE trials of ``study``, none when the database or the study is absent.

    Optuna is imported here, not at module scope, so the generator runs with no study
    database present, which is the state the checked-in catalogues were generated in.
    """
    if not db.exists():
        return []
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        loaded = optuna.load_study(study_name=study, storage=f"sqlite:///{db}")
    except KeyError:  # no such study
        return []
    complete = (optuna.trial.TrialState.COMPLETE,)
    return loaded.get_trials(deepcopy=False, states=complete)


def best_trial(db: Path, study: str, budget: int = TRIAL_BUDGET) -> Trial | None:
    """Best of the first ``budget`` completed trials of ``study``, or None when there is
    no such study or no completed trial.

    The window holds COMPLETE trials alone, in trial-number order, so failed, pruned and
    waiting trials and the orphan running rows of killed drivers take no place in it.
    """
    return _best_within_budget(_complete_trials(db, study), budget)


def overrides_of(params: dict[str, str]) -> list[str]:
    """The search-space parameters as hydra override tokens, sorted by key.

    Optuna parameter names are already the override paths (taken straight from
    ``hydra.sweeper.params``), so only the list values need quoting.
    """
    return [f"{k}={quote(params[k])}" for k in sorted(params)]


@dataclass(frozen=True)
class Layout:
    """Where one data set's and feature set's catalogues, studies and runs live.

    Each data set has trees of its own, named after it. Its default set
    (``DEFAULT_LABEL``) sits at the root of its trees under ``logs/``: the four-momentum
    that every hls4ml experiment config reads, and the [pt, etarel, phirel] that every
    top-tagging grid experiment sets. Any other set is held apart under its own label,
    the underscore join of its feature names that ``src/utils/results_table.py`` also
    uses for ``results/<label>/``. The catalogues of the grid rows go under
    ``<out>/<label>/``, the default set's included, and those of the rows that pin their
    columns at ``<out>/``.
    """

    features: tuple[str, ...] | None = None
    dataset: str = "hls4ml"

    @property
    def label(self) -> str | None:
        """The directory component of this feature set, None for the data set's default
        set, named or not, whose blocks read the experiment's own columns."""
        label = "_".join(self.features) if self.features else None
        return None if label == DEFAULT_LABEL[self.dataset] else label

    @property
    def budget(self) -> int:
        """The completed trials of a study within which its best is taken."""
        return TOP_TAGGING_BUDGET if self.dataset == "top_tagging" else TRIAL_BUDGET

    def under(self, root: str) -> str:
        """``root`` itself, or the subdirectory of it this feature set owns."""
        return f"{root}/{self.label}" if self.label else root

    def catalogue_dir(self, out: Path, model: str) -> Path:
        """Where a row's catalogues go: ``out``, or this set's directory under it."""
        if model in EMBEDDED or model in JET_LEVEL:
            return out
        return out / (self.label or DEFAULT_LABEL[self.dataset])

    def run_dir(self, task: str, model: str, nconst: int, label: str) -> str:
        """Where one block's hydra run directory goes under logs/."""
        base = self.under(f"logs/{task}/{self.dataset}")
        return f"{base}/{model}/nconst{nconst}/{label}"

    def overrides(self) -> list[str]:
        """The feature-set override every block of this layout carries, if any."""
        if self.label is None:
            return []
        return [f"data.features={quote('[' + ','.join(self.features) + ']')}"]


def _block(header: list[str], entry: str, body: list[str]) -> list[str]:
    """One commented block: banner, header lines, then the wrapped command."""
    lines = [SEP, *header, SEP, f"# python3 {entry} \\"]
    lines += [f"#     {tok} \\" for tok in body[:-1]] + [f"#     {body[-1]}", ""]
    return lines


def follows_recipe(model: str, layout: Layout) -> bool:
    """Whether a row takes its hyperparameters from a published recipe, never from a
    study."""
    return layout.dataset == "top_tagging" and model in TOP_TAGGING_RECIPES


def counts_of(model: str, layout: Layout) -> list[int]:
    """The constituent counts a row is trained at in this layout's data set."""
    if layout.dataset == "top_tagging":
        return TOP_TAGGING_NCONSTITUENTS
    return NCONSTITUENTS_OF[model]


def _provenance(trial: Trial | None, study: str, recipe: bool) -> str:
    """The header line saying where a block's hyperparameters came from."""
    if recipe:
        return "# published top-tagging recipe, from the experiment config"
    if trial is None:
        return "# transferred from experiment config (no study yet)"
    return f"# best trial {trial[0]} of {study}, val/acc_best={trial[1]:.6f}"


def _header(
    model: str, nconst: int, variant: Variant, trial: Trial | None, layout: Layout
) -> list[str]:
    """The two comment lines above a block: its grid cell, and its hyperparameters'
    provenance."""
    label, method, scheme = variant
    study = study_name(model, nconst, method, scheme, layout.dataset)
    return [
        f"# {model}  nconst={nconst}  {label}  (method={method}, group={scheme})",
        _provenance(trial, study, follows_recipe(model, layout)),
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
        f"experiment={layout.dataset}/{model}_{nconst}",
        f"data.method={method}",
        f"data.augmentation_scheme={scheme}",
        f"data.n_folds={N_FOLDS}",
        *layout.overrides(),
    ]
    return body + (overrides_of(trial[2]) if trial else [])


def trainer_overrides(model: str) -> list[str]:
    """The trainer a block selects: the CPU for a row of ``CPU_ROWS``, else GPU 0, which
    the submitters rewrite to the device they place the block on."""
    if model in CPU_ROWS:
        return ["trainer=cpu"]
    return ["trainer=gpu", "trainer.devices=[0]"]


def train_block(
    model: str, nconst: int, variant: Variant, trial: Trial | None, layout: Layout
) -> list[str]:
    """One training command, folds included, writing to its own hydra run dir."""
    body = _common_body(model, nconst, variant, trial, layout) + [
        *trainer_overrides(model),
        f"hydra.run.dir={layout.run_dir('train', model, nconst, variant[0])}",
    ]
    header = _header(model, nconst, variant, trial, layout)
    return _block(header, "src/train.py", body)


def eval_block(
    model: str, nconst: int, variant: Variant, trial: Trial | None, layout: Layout
) -> list[str]:
    """One evaluation command, reading the manifests its training block wrote.

    The path names the training's run directory, not its manifest file: src/eval.py
    then merges the main run's manifest with every fold_<i>/ one, where the file would
    test the main run alone.
    """
    label = variant[0]
    train_dir = layout.run_dir("train", model, nconst, label)
    body = _common_body(model, nconst, variant, trial, layout) + [
        f"checkpoint_manifest_path={train_dir}",
        "results_table.invariance_metric=tv",
        *trainer_overrides(model),
        f"hydra.run.dir={layout.run_dir('eval', model, nconst, label)}",
    ]
    header = _header(model, nconst, variant, trial, layout)
    return _block(header, "src/eval.py", body)


BASELINE_ONLY_NOTE = [
    "# Baseline only: this architecture embeds the symmetry rather than acquiring it,",
    "# so the augmentation and SCL regimes are not run on it.",
]
JET_LEVEL_NOTE = [
    "# Baseline only: the jet-level observables are computed upstream and never",
    "# transformed, so the augmentation and SCL regimes have nothing to act on.",
]
RECIPE_NOTE = [
    "# The block follows the published recipe its experiment config holds; no study",
    "# tunes it.",
]


def regime_note(model: str) -> list[str]:
    """Why a catalogue covers the baseline alone; nothing for the full grid."""
    if model in JET_LEVEL:
        return JET_LEVEL_NOTE
    return BASELINE_ONLY_NOTE if model in EMBEDDED else []


def tuned_note(model: str, layout: Layout) -> list[str]:
    """Where the blocks' hyperparameters were read from, for the reader of one file."""
    if follows_recipe(model, layout):
        return RECIPE_NOTE
    optuna_root = layout.under(f"logs/optuna/{layout.dataset}")
    return [
        f"# Each block carries the best of the first {layout.budget} completed trials, by trial",
        f"# number, of its own study under {optuna_root}/;",
        "# a block whose study is missing carries none and says so (transferred ablation).",
    ]


def features_note(layout: Layout) -> list[str]:
    """Which feature set a catalogue is for, and the trees it keeps to; none by
    default."""
    if layout.label is None:
        return []
    return [
        f"# Feature set [{', '.join(layout.features)}]: studies from"
        f" {layout.under(f'logs/optuna/{layout.dataset}')}/, runs under"
        f" logs/{{train,eval}}/{layout.dataset}/{layout.label}/."
    ]


def file_header(model: str, task: str, layout: Layout) -> list[str]:
    """The banner a catalogue opens with, naming the grid it covers and where it came
    from."""
    what = "TRAINING" if task == "train" else "EVALUATION"
    ncounts, nvariants = len(counts_of(model, layout)), len(VARIANTS_OF[model])
    grid = f"{ncounts} x {nvariants} = {ncounts * nvariants}"
    return [
        "#!/usr/bin/env bash",
        BANNER,
        f"# {model.upper()} {what} COMMANDS",
        BANNER,
        f"# One block per (constituent count, regime): {grid} {task} runs. Generated",
        "# by scripts/experiments/make_train_catalogues.py; regenerate"
        " rather than edit by hand.",
        *features_note(layout),
        *regime_note(model),
        *tuned_note(model, layout),
        "#",
        "# Every command is commented out: copy one to run it by hand, or feed the whole",
        f"# file to scripts/experiments/submit_{task}.sh, which queues them"
        " on a GPU or slurm.",
        "",
    ]


def _tuned(
    model: str, nconst: int, variant: Variant, dbs: Path, layout: Layout
) -> Trial | None:
    """The best trial a block carries; None for a row on a published recipe."""
    if follows_recipe(model, layout):
        return None
    study = study_name(model, nconst, variant[1], variant[2], layout.dataset)
    return best_trial(dbs / f"{model}.db", study, layout.budget)


def render(model: str, task: str, dbs: Path, layout: Layout) -> str:
    """The full text of one catalogue."""
    make_block = train_block if task == "train" else eval_block
    out = file_header(model, task, layout)
    for nconst in counts_of(model, layout):
        out += [BANNER, f"# {model}  nconst={nconst}", BANNER, ""]
        for variant in VARIANTS_OF[model]:
            trial = _tuned(model, nconst, variant, dbs, layout)
            out += make_block(model, nconst, variant, trial, layout)
    return "\n".join(out).rstrip("\n") + "\n"


def _show(path: Path, repo: Path) -> str:
    """Repo-relative when the path is inside the repository, absolute otherwise."""
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _emit(text: str, path: Path, repo: Path, dry_run: bool) -> int:
    """Write one catalogue, or report what writing it would do; 1 when it would change
    it."""
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {_show(path, repo)}")
        return 0
    same = path.exists() and path.read_text() == text
    print(f"{_show(path, repo)}: {'unchanged' if same else 'WOULD CHANGE'}")
    return 0 if same else 1


def models_of(layout: Layout) -> list[str]:
    """The rows a layout has catalogues for.

    Each embedded architecture pins its own columns through ``required_features``: the
    four-momentum for all but ``vn``, whose experiment configs set ``[pt, eta, phi]``,
    and the three moment rows, whose experiment configs set ``[pt, etarel, phirel]``.
    The jet-level rows read no constituent column at all. Only a layout without
    ``--features`` has catalogues for either. The top-tagging data set has its own rows.
    """
    rows = TOP_TAGGING if layout.dataset == "top_tagging" else [*MODELS, *JET_LEVEL]
    if layout.features is None:
        return rows
    return [model for model in rows if model in MODELS and model not in EMBEDDED]


def _generate(dbs: Path, out: Path, repo: Path, layout: Layout, dry_run: bool) -> int:
    """Write or check every catalogue; returns how many differ from a fresh
    generation."""
    stale = 0
    for model in models_of(layout):
        for task in ("train", "eval"):
            text = render(model, task, dbs, layout)
            path = layout.catalogue_dir(out, model) / f"run{model}_{task}.sh"
            stale += _emit(text, path, repo, dry_run)
    return stale


def _parse_args() -> argparse.Namespace:
    """Command line: which data set, where the studies are, where the catalogues go, and
    what to write."""
    # Five flat add_argument declarations: helpers would only scatter the CLI.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="hls4ml",
        help="The data set whose catalogues are written (default hls4ml).",
    )
    parser.add_argument(
        "--dbs",
        default=None,
        help="Study databases directory (default <repo>/logs/optuna/<dataset>"
        "[/<feature label>]).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where the catalogues go, the grid rows' under <feature label>/ (default "
        "<repo>/scripts/experiments/<dataset>).",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature set of the grid rows, e.g. 'pt,eta,phi'; without "
        "it, every row on the data set's default set (px,py,pz,e for hls4ml, "
        "pt,etarel,phirel for top_tagging).",
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
    repo = Path(__file__).resolve().parents[2]
    layout = Layout(features=_features(args.features), dataset=args.dataset)
    dbs = _default(args.dbs, repo / "logs" / "optuna" / args.dataset, layout)
    out = Path(args.out or repo / "scripts" / "experiments" / args.dataset).expanduser()
    stale = _generate(dbs, out, repo, layout, args.dry_run)
    if stale:
        print(f"\n{stale} file(s) differ from a fresh generation.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
