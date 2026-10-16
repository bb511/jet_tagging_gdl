"""Learning-rate schedules of the Keras rows, which their optimiser steps per batch.

The schedules of the torch rows are in schedules.py. This module imports Keras, which
the optional Poetry group ``hgq`` installs.
"""

import math
from typing import Any

# The component selects Keras's JAX backend before Keras is first imported.
from src.algorithms.components import hgq_port  # noqa: F401

# isort: split

import keras


@keras.saving.register_keras_serializable(package="jet_tagging_gdl")
class CosineRestarts(keras.optimizers.schedules.LearningRateSchedule):
    """A cosine decay restarted ``cycles`` times over the run, each cycle ending on its
    floor for its last ``flat_fraction``: the schedule of JEDI-linear's training
    (arXiv:2508.15468, ``cosine_decay_restarts`` of its ``src/train.py``), stepped per
    batch where theirs is set per epoch.

    ``configure_optimizers`` builds it as it builds ``CosineDecay``: ``decay_steps`` are
    the steps of the whole run and ``alpha`` is the floor as a fraction of
    ``initial_learning_rate``.
    """

    KEYS = ("initial_learning_rate", "decay_steps", "alpha", "cycles", "flat_fraction")

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        alpha: float = 0.0,
        cycles: int = 1,
        flat_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha, self.cycles, self.flat_fraction = alpha, cycles, flat_fraction

    def __call__(self, step: Any) -> Any:
        start = keras.ops.convert_to_tensor(self.initial_learning_rate)
        cycle = self.decay_steps / self.cycles
        position = keras.ops.mod(keras.ops.cast(step, start.dtype), cycle)
        t = keras.ops.minimum(position / (cycle * (1.0 - self.flat_fraction)), 1.0)
        floor = self.alpha * start
        return floor + (start - floor) * 0.5 * (1.0 + keras.ops.cos(math.pi * t))

    def get_config(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.KEYS}
