# One-vs-rest ROC curves, one panel per jet class.
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from src.plot import save_figure


def plot(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    save_dir: Path,
    split: str = "test",
) -> Path:
    """Plot the one-vs-rest ROC curve of every jet class.

    :param labels: true class index per jet, shape (jets,).
    :param probs: class probabilities, shape (jets, classes), columns ordered as
        *class_names*.
    :param split: the split the scores come from; it names the file and the title.
    """
    plt.style.use(hep.style.CMS)
    n_classes = probs.shape[1]
    names = list(class_names)[:n_classes]
    one_vs_rest = label_binarize(labels, classes=list(range(n_classes)))
    fig, axes = plt.subplots(
        1,
        n_classes,
        figsize=(5 * n_classes, 5),
        dpi=120,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column, name in enumerate(names):
        _draw(axes[0][column], one_vs_rest[:, column], probs[:, column], name)
    axes[0][0].set_ylabel("True positive rate")
    fig.suptitle(f"One-vs-rest ROC, {split} set")
    fig.tight_layout()
    return save_figure(fig, save_dir, f"roc_curves_{split}")


def _draw(ax: Axes, truth: np.ndarray, score: np.ndarray, name: str) -> None:
    """Draw one class against all the others, with its AUC in the legend."""
    false_positive, true_positive, _ = roc_curve(truth, score)
    area = auc(false_positive, true_positive)
    ax.plot(false_positive, true_positive, linewidth=2, label=f"AUC = {area:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(name)
    ax.set_xlabel("False positive rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
