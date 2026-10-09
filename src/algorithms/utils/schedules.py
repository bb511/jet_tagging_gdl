"""Learning-rate schedules of the published top-tagging recipes, as factors of the
initial rate stepped once per batch.

Each factory takes ``total_steps``, which `JetTaggingModule._set_up_scheduler` injects
once the trainer knows the run's length, and is referenced from a config as a partial,
``{_target_: src.algorithms.utils.schedules.<name>, _partial_: true}``, beside
``interval: step``.
"""

import math
from dataclasses import dataclass
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _half_cosine(t: float) -> float:
    """1 at t = 0, falling to 0 at t = 1."""
    return 0.5 * (1.0 + math.cos(math.pi * t))


def _cycle(t: float, t0: float, t_mult: float) -> float:
    """The fraction of its cycle that *t* has run through, the cycles of
    `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` lasting t0, t0 t_mult, t0
    t_mult^2 and so on."""
    length = t0
    while t >= length:
        t, length = t - length, length * t_mult
    return t / length


def _cosine(step: int, total_steps: int, eta_min: float) -> float:
    return eta_min + (1.0 - eta_min) * _half_cosine(min(step / total_steps, 1.0))


def cosine(optimizer: Optimizer, total_steps: int, eta_min: float = 0.0) -> LambdaLR:
    """Anneal to *eta_min* times the initial rate over the run (L-GATr-slim)."""
    factor = partial(_cosine, total_steps=total_steps, eta_min=eta_min)
    return LambdaLR(optimizer, factor)


def _restarts(step: int, length: float) -> float:
    """The factor *step* steps in, the cycles lasting *length* steps."""
    return _half_cosine(_cycle(step, length, 1.0))


def cosine_restarts(
    optimizer: Optimizer, total_steps: int, cycles: int = 2
) -> LambdaLR:
    """Anneal to zero *cycles* times over the run, restarting at the initial rate, as
    the LLoCa recipe does with two cycles of 150 000 steps."""
    return LambdaLR(optimizer, partial(_restarts, length=total_steps / cycles))


@dataclass(frozen=True)
class _WarmupRestartsDecay:
    """The factor of `warmup_restarts_decay` at one step; the times are in epochs."""

    steps_per_epoch: float
    epochs: int
    warmup: float
    t0: float
    t_mult: float
    decay_epochs: int
    gamma: float

    def __call__(self, step: int) -> float:
        """The factor at *step*, counted from 0 at the first batch."""
        epoch = step / self.steps_per_epoch
        decay_start = self.epochs - self.decay_epochs
        if epoch < self.warmup:
            return epoch / self.warmup
        if epoch < decay_start:
            return self._restarts(epoch)
        decays = math.floor(epoch - decay_start) + 1
        return self._restarts(decay_start - 1) * self.gamma**decays

    def _restarts(self, epoch: float) -> float:
        return _half_cosine(_cycle(epoch - self.warmup, self.t0, self.t_mult))


def warmup_restarts_decay(
    optimizer: Optimizer,
    total_steps: int,
    epochs: int,
    warmup: float = 4,
    t0: float = 4,
    t_mult: float = 2,
    decay_epochs: int = 3,
    gamma: float = 0.5,
) -> LambdaLR:
    """The LorentzNet and PELICAN schedule: a linear warm-up over *warmup* epochs,
    cosine annealing with warm restarts of *t0* epochs growing by *t_mult*, then
    *decay_epochs* epochs of decay by *gamma* (arXiv:2201.08187, section 3.3;
    arXiv:2211.00454, section 4).

    The decay starts from the rate at the start of the last cosine epoch and multiplies
    it by *gamma* once per epoch, as the LorentzNet reference, whose scheduler steps
    once per epoch, does (LorentzNet-release, ``top_tagging.py``); the warm-up and the
    cosine are evaluated at every step.

    :param epochs: the run's length, ``trainer.max_epochs``, which splits *total_steps*
        into epochs.
    """
    schedule = _WarmupRestartsDecay(
        total_steps / epochs, epochs, warmup, t0, t_mult, decay_epochs, gamma
    )
    return LambdaLR(optimizer, schedule)
