"""Figures written to disk, one family per module.

Every module exposes a single ``plot`` function, from the numbers and a directory to the
path it wrote. No module keeps state, so a callback may call them from any process.
"""

import matplotlib

# Chosen before any module imports pyplot: the runs are headless (ssh, nohup, batch jobs)
# and the backend cannot be swapped once pyplot has been imported.
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_figure(fig: Figure, save_dir: Path, name: str) -> Path:
    """Write a figure to ``<save_dir>/<name>.png``, creating the directory, and release
    it."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
