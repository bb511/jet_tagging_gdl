# Plot families drawn by the callbacks, one family per module.
"""Figures written to disk, one family per module.

Every module exposes a single ``plot`` function that takes the numbers, a directory, and returns
the path it wrote. Nothing is shown on screen and no module keeps state, so a callback may call
them from any process.
"""

import matplotlib

# Chosen before any module imports pyplot: the runs are headless (ssh, nohup, batch jobs)
# and the backend cannot be swapped once pyplot has been imported.
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathvalidate import sanitize_filename


def save_figure(fig: Figure, save_dir: Path, name: str) -> Path:
    """Write a figure to ``<save_dir>/<name>.png`` and release it.

    :param save_dir: created if it does not exist.
    :param name: file stem, sanitised for the file system.
    :return: the path written.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{sanitize_filename(name).replace(' ', '_')}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.clear()
    plt.close(fig)
    return path
