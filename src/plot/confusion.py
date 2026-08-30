# Confusion matrix: the raw counts beside their row-normalised version.
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from src.plot import save_figure


def plot(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
    save_dir: Path,
    split: str = "test",
) -> Path:
    """Plot the confusion matrix of one pass, as counts and as fractions.

    :param labels: true class index per jet, shape (jets,).
    :param preds: predicted class index per jet, shape (jets,).
    :param class_names: display names of the classes, in label order.
    :param save_dir: directory the PNG is written to.
    :param split: the split the predictions come from; it names the file and the title.
    :return: the path written.
    """
    plt.style.use(hep.style.CMS)
    names = list(class_names)
    counts = confusion_matrix(labels, preds, labels=list(range(len(names))))
    # Rows are the true class, so a row-normalised entry is that class's recall. The clip
    # keeps a class absent from the split from dividing by zero.
    fractions = counts.astype(float) / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    _draw(fig, axes[0], counts, names, "counts", "d", None)
    _draw(fig, axes[1], fractions, names, "row-normalised", ".2f", (0.0, 1.0))
    fig.suptitle(f"Confusion matrix, {split} set")
    fig.tight_layout()
    return save_figure(fig, save_dir, f"confusion_matrix_{split}")


def _draw(
    fig: Figure,
    ax: Axes,
    data: np.ndarray,
    names: list[str],
    title: str,
    fmt: str,
    limits: tuple[float, float] | None,
) -> None:
    """Draw one heatmap, its axes labelled by class and its cells annotated."""
    vmin, vmax = limits if limits else (None, None)
    image = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.set_xticks(range(len(names)), labels=names, rotation=45)
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _annotate(ax, data, fmt)


def _annotate(ax: Axes, data: np.ndarray, fmt: str) -> None:
    """Write every cell's value, in white where the colour map is too dark for black."""
    threshold = data.max() / 2.0
    for row, column in np.ndindex(data.shape):
        ax.text(
            column,
            row,
            format(data[row, column], fmt),
            ha="center",
            va="center",
            color="white" if data[row, column] > threshold else "black",
            fontsize=9,
        )
