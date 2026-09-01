# Training curves: loss, accuracy and learning rate against the epoch.
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.axes import Axes

from src.plot import save_figure


def plot(history: dict[str, list[float]], save_dir: Path) -> Path:
    """Plot the training history of one run as three panels.

    :param history: the ``epoch``, ``train/loss``, ``train/acc``, ``val/loss``, ``val/acc`` and
        ``lr`` series, one entry per training epoch and NaN where a metric was not logged.
    """
    plt.style.use(hep.style.CMS)
    epochs = np.asarray(history["epoch"])
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=120)
    _draw(axes[0], epochs, {"train": history["train/loss"], "val": history["val/loss"]}, "Loss")
    _draw(axes[1], epochs, {"train": history["train/acc"], "val": history["val/acc"]}, "Accuracy")
    _draw_lr(axes[2], epochs, np.asarray(history["lr"], dtype=float))
    for ax in axes:
        ax.set_xlabel("Epoch")
        # Epochs are integers, and a run of a few of them would otherwise get half ticks.
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    return save_figure(fig, save_dir, "training_curves")


def _draw(ax: Axes, epochs: np.ndarray, series: dict[str, list[float]], ylabel: str) -> None:
    """Draw one panel, every series of *series* against the epoch."""
    for label, values in series.items():
        ax.plot(epochs, values, "o-", label=label, linewidth=2, markersize=4)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _draw_lr(ax: Axes, epochs: np.ndarray, learning_rate: np.ndarray) -> None:
    """Draw the learning rate, logarithmic since every schedule here decays by factors."""
    if not np.all(np.isnan(learning_rate)):
        ax.plot(epochs, learning_rate, "o-", color="green", linewidth=2, markersize=4)
        ax.set_yscale("log")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
