#!/usr/bin/env python3
"""Print the top-tagging figures of evaluated runs beside the published ones.

Each argument is the run directory of a top-tagging evaluation by ``src/eval.py``, whose
``outputs/main.npz`` and ``outputs/fold_<i>.npz`` hold the clean logits (jets, 2), column
0 qcd and column 1 top, and the class index of each jet. For each directory the figures
of the main run are printed, and below them their mean ± standard deviation over the
folds: the accuracy, the AUC of top against qcd, and the background rejection 1/eps_B at
a signal efficiency eps_S of 30 % and of 50 %, top being the signal. The published
figures follow, each with the arXiv number of the paper it was read from.

    python3 scripts/analysis/top_tagging_summary.py \\
        logs/eval/top_tagging/lorentznet/nconst100/baseline \\
        logs/eval/top_tagging/deepsets/nconst100/baseline
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

QCD, TOP = 0, 1
# The signal efficiency of each rejection.
REJECTIONS = {"rejection_es30": 0.3, "rejection_es50": 0.5}
# (figure, decimals printed), in the column order of the published tables.
COLUMNS = (("acc", 4), ("auc", 4), ("rejection_es30", 0), ("rejection_es50", 0))
ROW = "{:<24} {:<12} {:>16} {:>16} {:>12} {:>12}"
HEADER = ("tagger", "source", "accuracy", "AUC", "1/eB(0.3)", "1/eB(0.5)")
# (tagger, arXiv number, accuracy, AUC, 1/eps_B at eps_S = 30 % and 50 %) as published;
# "-" where the paper reports none.
LITERATURE = (
    # Table 1, the mean of 6 runs.
    ("LorentzNet", "2201.08187", "0.942", "0.9868", "2195 ± 173", "498 ± 18"),
    # Table 2, Msg 60 / Agg 35, five seeds.
    (
        "PELICAN 48k",
        "2307.16506",
        "0.9424 ± 0.0001",
        "0.9868 ± 0.0001",
        "2148 ± 125",
        "-",
    ),
    # Table 1, five trainings.
    ("L-GATr-slim", "2512.17011", "0.9420", "0.9869 ± 0.0001", "2264 ± 93", "546 ± 7"),
    # Table 5, five seeds.
    (
        "LLoCa-Transformer",
        "2508.14898",
        "0.9416 ± 0.0001",
        "0.9866 ± 0.0001",
        "2150 ± 130",
        "492 ± 15",
    ),
    # Trained from scratch on the set.
    ("ParT", "2202.03772", "0.940", "0.9858", "1602 ± 81", "413 ± 16"),
    # The taggers on expert variables of the 2019 comparison, Table 1, single training.
    ("N-subjettiness 6 (DNN)", "1902.09914", "0.922", "0.979", "792 ± 18", "-"),
    ("N-subjettiness 8 (DNN)", "1902.09914", "0.929", "0.981", "867 ± 15", "-"),
    ("LDA", "1902.09914", "0.892", "0.955", "151 ± 0.4", "-"),
    ("EFP (linear)", "1902.09914", "0.932", "0.980", "384", "-"),
    # As PELICAN's Table 1 quotes them: BIP is Munoz et al., Mach. Learn.: Sci. Technol. 3
    # (2022) 04LT05, and DisCo-FFS arXiv:2212.00046 by PELICAN's reference list.
    ("BIP (XGBoost)", "2307.16506", "0.929", "0.978", "600 ± 47", "-"),
    ("BIP (MLP)", "2307.16506", "0.931", "0.981", "853 ± 68", "-"),
    ("DisCo-FFS", "2307.16506", "-", "0.982", "1249 ± 43", "-"),
)


def rejection(score: np.ndarray, top: np.ndarray, efficiency: float) -> float:
    """1/eps_B at the threshold that at most *efficiency* of the top jets pass, read as
    ``src/callbacks/invariance.py`` reads its working points.

    The threshold is the stored top score at the quantile 1 - efficiency, the jets
    strictly above it pass, and a point no qcd jet passes reads n_B, the passing count
    being floored at one jet.
    """
    threshold = np.quantile(score[top], 1.0 - efficiency, method="higher")
    passing = np.count_nonzero(score[~top] > threshold)
    return np.count_nonzero(~top) / max(passing, 1)


def figures(path: Path) -> dict[str, float]:
    """The figures of one ``.npz``, computed in float64.

    The score is the log-odds of top, to which the one-vs-rest score of the callback
    reduces for two classes. The accuracy is the micro one the papers quote, where the
    results table's test_acc averages the two class recalls.
    """
    with np.load(path) as outputs:
        logits = outputs["logits_clean"].astype(np.float64)
        labels = outputs["labels"]
    top, score = labels == TOP, logits[:, TOP] - logits[:, QCD]
    return {
        "acc": float(np.mean(logits.argmax(axis=1) == labels)),
        "auc": float(roc_auc_score(top, score)),
        **{name: rejection(score, top, e) for name, e in REJECTIONS.items()},
    }


def run_figures(run_dir: Path) -> tuple[dict[str, float], list[dict[str, float]]]:
    """The figures of a run's main run, and of each of its folds in fold order."""
    outputs = run_dir / "outputs"
    folds = sorted(
        outputs.glob("fold_*.npz"), key=lambda p: int(p.stem.removeprefix("fold_"))
    )
    return figures(outputs / "main.npz"), [figures(path) for path in folds]


def _cell(values: list[float], decimals: int) -> str:
    """One value, or the mean ± sample standard deviation of several, as the results
    table summarises its folds."""
    if len(values) == 1:
        return f"{values[0]:.{decimals}f}"
    mean, std = np.mean(values), np.std(values, ddof=1)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _cells(runs: list[dict[str, float]]) -> list[str]:
    return [_cell([run[name] for run in runs], decimals) for name, decimals in COLUMNS]


def _label(run_dir: Path) -> str:
    """The row an evaluation belongs to: ``<row>`` in the catalogues' layout
    ``<row>/nconst<N>/<regime>``, the directory's own name otherwise."""
    if run_dir.parent.name.startswith("nconst"):
        return run_dir.parent.parent.name
    return run_dir.name


def _rows(run_dir: Path) -> list[tuple[str, ...]]:
    """The table rows of one evaluation: its main run, then its folds if it has any."""
    main, folds = run_figures(run_dir)
    label = _label(run_dir)
    rows = [(label, "main", *_cells([main]))]
    if folds:
        rows.append((label, f"folds ({len(folds)})", *_cells(folds)))
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dirs", nargs="+", type=Path, help="Evaluation run directories."
    )
    return parser.parse_args()


def main() -> None:
    """Print one table: the evaluated runs, then the published figures."""
    args = _parse_args()
    print(ROW.format(*HEADER))
    for run_dir in args.run_dirs:
        for row in _rows(run_dir):
            print(ROW.format(*row))
    print()
    for row in LITERATURE:
        print(ROW.format(*row))


if __name__ == "__main__":
    main()
