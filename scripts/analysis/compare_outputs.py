#!/usr/bin/env python3
"""Compare the clean test outputs of two evaluations of the same jets, figure by figure.

The inputs are two files in the layout of ``src/eval.py``'s ``outputs/<label>.npz``: a
reference (a float network, or a quantised one) and a candidate (a quantised network, or
its alkaid design as ``scripts/synthesis/synthesise_vn.py`` writes it), each holding
``logits_clean`` (jets, classes) and ``labels`` (jets,) of the same test jets in the
same order; the labels must agree. Each is scored on its own, per class and as the macro
mean over the classes: the accuracy, the one-vs-rest AUC and the clean figures of the
three working points of `src.symmetry.WORKING_POINTS` (the rejection 1/eps_B at eps_S =
30 %, the TPR at FPR = 1 % and at FPR = 10 %). Every figure of the candidate is its own,
read at thresholds set on its own scores, as the results table reports each row; none is
read at the reference's thresholds, as the retentions of the invariance probe are. The
working points and the AUC come from `src.callbacks.invariance.clean_metrics`, so they
equal the ``performance/*`` keys an evaluation logs. The accuracy of a class is its
recall, and their macro mean is torchmetrics' ``MulticlassAccuracy``, the table's
``test_acc``.

Beside each pair of figures stands their relative difference, (candidate - reference) /
reference. ``--tolerance`` (1 % by default) marks a working point PASS when that
difference is at most the tolerance in magnitude, either way, and the verdict is PASS
when the three macro working points all are. Per class, the largest vertical gap between
the two ROC curves is reported beside them and judged by no tolerance: the maximum of
abs(TPR_candidate - TPR_reference) over the FPR f at or above ``--fpr-floor`` (1e-3 by
default), each TPR being that of the last threshold at an FPR of at most f. Both are
step functions of f, so the maximum over the floor and the FPR of every threshold of
either curve above it is exact. Below the floor, where a curve rests on fewer than
floor * n_B of the n_B background jets of its class and climbs steepest, a single
background jet that changes rank opens a step 1/n_B wide in FPR and as high as the share
of the signal jets it passes; the floor keeps such single-jet steps out of the gap.

    python3 scripts/analysis/compare_outputs.py \\
        --reference <float eval dir>/outputs/main.npz \\
        --candidate <quantised eval dir>/outputs/main.npz \\
        --label <run> --csv <table>.csv --figure <roc>.pdf

``--csv`` appends one row named by ``--label``, so that a loop over the evaluations of a
grid builds one comparison table; a file with other columns is refused. ``--figure``
draws the ROC curves of the two evaluations, one panel per class, on a logarithmic FPR
axis.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any

import common
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve

# common puts the repository root on the import path, which the src imports below need.
from src.callbacks.invariance import clean_metrics, one_vs_rest
from src.symmetry import (
    CLASS_NAMES,
    PERFORMANCE_FAMILIES,
    WORKING_POINTS,
    clean_family,
    key,
)

# The accuracy of a class is its recall.
ACCURACY = "acc"
FAMILIES = (ACCURACY, *PERFORMANCE_FAMILIES)
# The families the tolerance judges.
JUDGED = tuple(clean_family(point) for point in WORKING_POINTS)
# Per family, the macro mean (None), then each class; printed the other way round.
SCOPES = (None, *CLASS_NAMES)
PRINTED = (*CLASS_NAMES, None)
FIELDS = ("reference", "candidate", "rel")
# Points drawn per curve, log-spaced in FPR: every threshold of the full test split
# would make a PDF of megabytes.
DRAWN = 2000

# The (fpr, tpr) of every threshold of one ROC curve, both non-decreasing, starting at
# (0, 0).
Curve = tuple[np.ndarray, np.ndarray]

OPTIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("--reference", {"type": Path, "required": True, "help": "the reference .npz"}),
    ("--candidate", {"type": Path, "required": True, "help": "the compared .npz"}),
    (
        "--tolerance",
        {"type": float, "default": 0.01, "help": "largest relative difference passed"},
    ),
    (
        "--fpr-floor",
        {"type": float, "default": 1e-3, "help": "least FPR the ROC gap reads"},
    ),
    ("--label", {"help": "names the CSV row (default: the candidate's path)"}),
    ("--csv", {"type": Path, "help": "CSV the row is appended to"}),
    ("--figure", {"type": Path, "help": "ROC overlays, format from the suffix"}),
)


def read_outputs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The clean logits (jets, classes) and the labels (jets,) of one evaluation.

    :raises ValueError: when a logit is not finite, which leaves no figure defined.
    """
    with np.load(path) as arrays:
        logits, labels = arrays["logits_clean"], arrays["labels"].astype(np.int64)
    if not np.isfinite(logits).all():
        raise ValueError(f"{path} holds non-finite logits.")
    return logits, labels


def accuracies(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """The recall of each class and their macro mean, as torchmetrics'
    ``MulticlassAccuracy`` takes it: a class neither present nor predicted is left out,
    and one predicted but absent counts as zero."""
    classes = np.arange(logits.shape[1])[:, None]
    predicted, true = logits.argmax(axis=1) == classes, labels == classes
    hits, support = (predicted & true).sum(axis=1), true.sum(axis=1)
    recall = np.divide(hits, support, out=np.zeros(len(classes)), where=support > 0)
    macro = float(recall[(predicted | true).any(axis=1)].mean())
    per_class = zip(CLASS_NAMES, recall, strict=True)
    return {key(None, ACCURACY): macro} | {
        key(None, ACCURACY, name): float(value) for name, value in per_class
    }


def scores(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Every figure of one evaluation, keyed as the evaluation logs them."""
    return accuracies(logits, labels) | clean_metrics(logits, labels)


def roc(signal: np.ndarray, score: np.ndarray) -> Curve:
    """The ROC curve of one class, every threshold kept, so that no knot of its step
    function is lost."""
    fpr, tpr, _ = roc_curve(signal, score, drop_intermediate=False)
    return fpr, tpr


def roc_curves(logits: np.ndarray, labels: np.ndarray) -> list[Curve]:
    """Per class, the ROC curve of the one-vs-rest scores that `clean_metrics` reads."""
    score = one_vs_rest(logits)
    return [roc(labels == c, score[:, c]) for c in range(score.shape[1])]


def tpr_at(curve: Curve, fpr: np.ndarray) -> np.ndarray:
    """The TPR of *curve* at an FPR of at most *fpr*: that of its last threshold there,
    which is the highest."""
    knots, tpr = curve
    return tpr[np.searchsorted(knots, fpr, side="right") - 1]


def roc_gap(reference: Curve, candidate: Curve, floor: float = 0.0) -> float:
    """The largest abs(TPR_candidate - TPR_reference) at equal FPR f >= *floor*.

    Both TPRs are step functions of f that change only at the FPR of a threshold of
    either curve, so the maximum over *floor* and every such FPR above it is exact.
    """
    fpr = np.union1d(np.union1d(reference[0], candidate[0]), [floor])
    fpr = fpr[fpr >= floor]
    return float(np.abs(tpr_at(candidate, fpr) - tpr_at(reference, fpr)).max())


def relative(candidate: float, reference: float) -> float:
    """(candidate - reference) / reference, NaN against a zero reference."""
    return (candidate - reference) / reference if reference else math.nan


def column(family: str, scope: str | None, field: str) -> str:
    """The CSV column of *field* of a family on one class, or on the macro mean when
    *scope* is None."""
    return f"{family}_{scope}_{field}" if scope else f"{family}_{field}"


def figure_columns(
    reference: dict[str, float], candidate: dict[str, float], tolerance: float
) -> dict[str, object]:
    """Per family and scope, both figures and their relative difference, then, for a
    working point, whether that difference is within *tolerance*."""
    columns: dict[str, object] = {}
    for family, scope in itertools.product(FAMILIES, SCOPES):
        name = key(None, family, scope)
        pair = (reference[name], candidate[name])
        values = dict(zip(FIELDS, (*pair, relative(pair[1], pair[0]))))
        columns |= {column(family, scope, f): v for f, v in values.items()}
        if family in JUDGED:
            columns[column(family, scope, "pass")] = abs(values["rel"]) <= tolerance
    return columns


def gap_columns(
    reference: list[Curve], candidate: list[Curve], floor: float
) -> dict[str, float]:
    """The largest ROC gap above the FPR *floor* over the classes, then that of each
    class."""
    gaps = [roc_gap(r, c, floor) for r, c in zip(reference, candidate, strict=True)]
    per_class = {f"roc_gap_{name}": gap for name, gap in zip(CLASS_NAMES, gaps)}
    return {"roc_gap_max": max(gaps)} | per_class


def failed(row: dict[str, object]) -> list[str]:
    """The working points whose macro figure is outside the tolerance."""
    return [family for family in JUDGED if not row[column(family, None, "pass")]]


def _head(args: argparse.Namespace, n_jets: int, passed: bool) -> dict[str, object]:
    """The columns that name the comparison, then its verdict."""
    return {
        "label": args.label or str(args.candidate),
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "n_jets": n_jets,
        "tolerance": args.tolerance,
        "fpr_floor": args.fpr_floor,
        "verdict": "PASS" if passed else "FAIL",
    }


def compare(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[Curve], list[Curve]]:
    """The comparison's row and the ROC curves of the reference and the candidate."""
    outputs = [read_outputs(path) for path in (args.reference, args.candidate)]
    labels = outputs[0][1]
    if not np.array_equal(labels, outputs[1][1]):
        raise SystemExit("The labels differ: the two files hold other jets or orders.")
    figures = figure_columns(*(scores(*pair) for pair in outputs), args.tolerance)
    curves = [roc_curves(*pair) for pair in outputs]
    head = _head(args, len(labels), not failed(figures))
    return head | figures | gap_columns(*curves, args.fpr_floor), *curves


def _cell(field: str, value: object) -> str:
    """One printed value: a flag, a signed percentage or five significant digits."""
    if field == "pass":
        return "PASS" if value else "FAIL"
    return f"{value:+.3%}" if field == "rel" else f"{value:.5g}"


def _line(name: str, cells: list[str]) -> str:
    return f"{name:<26}" + "".join(f"{cell:>11}" for cell in cells)


def print_table(row: dict[str, object]) -> None:
    """One line per family and field, one column per class and the macro mean last."""
    print(f"[load] {row['n_jets']} jets: {row['reference']} vs {row['candidate']}")
    print(_line("", [*CLASS_NAMES, "macro"]))
    for family in FAMILIES:
        fields = (*FIELDS, "pass") if family in JUDGED else FIELDS
        for field in fields:
            cells = [row[column(family, scope, field)] for scope in PRINTED]
            print(_line(f"{family} {field}", [_cell(field, c) for c in cells]))
    gaps = [f"{row[f'roc_gap_{name}']:.4g}" for name in CLASS_NAMES]
    print(_line("roc_gap (macro: max)", [*gaps, f"{row['roc_gap_max']:.4g}"]))


def print_verdict(row: dict[str, object]) -> None:
    """PASS, or FAIL with the working points outside the tolerance, then the largest ROC
    gap above the FPR floor, which no tolerance judges."""
    within = f"within {row['tolerance']:.2%} of the reference's"
    outside = failed(row)
    if outside:
        print(f"[verdict] FAIL: macro {', '.join(outside)} not {within}")
    else:
        print(f"[verdict] PASS: every macro working point {within}")
    gap = f"{row['roc_gap_max']:.4g} at FPR >= {row['fpr_floor']:g}"
    print(f"[verdict] largest ROC gap {gap}, not judged")


def _check_header(path: Path, fields: list[str]) -> None:
    """Refuse to append under a header with other columns, which would misalign the
    row."""
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    if header not in (None, fields):
        raise SystemExit(f"{path} holds other columns than a comparison; name another.")


def append_row(path: Path, row: dict[str, object]) -> None:
    """Append *row*, writing the header first into a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        _check_header(path, list(row))
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def _draw_class(ax: Axes, name: str, curves: tuple[Curve, Curve], gap: float) -> None:
    """Overlay the two curves of one class, from the smallest non-zero FPR of the
    reference to 1."""
    knots = curves[0][0]
    fpr = np.geomspace(knots[knots > 0].min(), 1.0, DRAWN)
    for label, curve in zip(("reference", "candidate"), curves):
        ax.plot(fpr, tpr_at(curve, fpr), drawstyle="steps-post", label=label)
    ax.set(xscale="log", xlabel="False positive rate", title=f"{name}, gap {gap:.3g}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def draw_rocs(
    path: Path, reference: list[Curve], candidate: list[Curve], row: dict[str, object]
) -> None:
    """One panel per class, the reference's and the candidate's ROC curves overlaid."""
    plt.style.use(hep.style.CMS)
    n = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    for ax, name, curves in zip(axes[0], CLASS_NAMES, zip(reference, candidate)):
        _draw_class(ax, name, curves, row[f"roc_gap_{name}"])
    axes[0][0].set_ylabel("True positive rate")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    args: argparse.Namespace,
    row: dict[str, object],
    curves: tuple[list[Curve], list[Curve]],
) -> None:
    """Append the row to ``--csv`` and draw ``--figure``, each when named."""
    if args.csv:
        append_row(args.csv, row)
        print(f"[write] row {row['label']} -> {common.shorten(args.csv.resolve())}")
    if args.figure:
        draw_rocs(args.figure, *curves, row)
        print(f"[write] ROC curves -> {common.shorten(args.figure.resolve())}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line: the two evaluations, the tolerance, the FPR floor and what to
    write."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for flag, options in OPTIONS:
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    if not 0.0 <= args.fpr_floor < 1.0:
        parser.error(f"--fpr-floor must lie in [0, 1), not {args.fpr_floor}")
    return args


def main(argv: list[str] | None = None) -> int:
    """Compare two evaluations, print the table and the verdict, write what is asked."""
    args = parse_args(argv)
    row, reference, candidate = compare(args)
    print_table(row)
    print_verdict(row)
    write_outputs(args, row, (reference, candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
