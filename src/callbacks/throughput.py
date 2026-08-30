# Throughput of the training loop: how fast the epochs run and how busy the GPU is.
"""Log the wall-clock cost of each training epoch.

The metrics answer whether a run is limited by the GPU or by the input pipeline, which is the
question an augmentation run raises: its transforms run inside the collate function, in the worker
processes, where the training step cannot time them. They surface instead as the gap between one
batch ending and the next beginning, ``perf/avg_data_ms``, which is comparable with the pure
input/output wait of a symmetry-consistency run.

Logged under ``perf/``: ``train_epoch_s`` (wall time of the epoch), ``train_steps_per_sec`` and
``train_samples_per_sec``, ``avg_step_ms`` (mean time in a training step, the forward and backward
pass without the data wait), ``avg_data_ms``, ``gpu_util_pct`` and ``gpu_mem_alloc_gb``, both
sampled once at the end of the epoch.
"""

import time

import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _gpu_utilization() -> float:
    """Percentage of the last sampling period the GPU was busy.

    Zero on a CPU run, NaN when the driver cannot be queried.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.utilization())
    except Exception:
        # torch.cuda.utilization() needs pynvml, which is absent on some driver installs.
        return float("nan")


class ThroughputCallback(Callback):
    """Time every training epoch and log its throughput through the module's logger."""

    def __init__(self) -> None:
        super().__init__()
        self._step_start: float = 0.0
        self._batch_end: float = 0.0
        self._epoch_start: float = 0.0
        self._step_times: list[float] = []
        self._data_times: list[float] = []
        self._epoch_samples: int = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the epoch's accumulators; the clock runs from here."""
        self._step_times = []
        self._data_times = []
        self._epoch_samples = 0
        self._batch_end = 0.0
        self._epoch_start = time.perf_counter()

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int
    ) -> None:
        """Charge the gap since the last batch ended to the input pipeline."""
        now = time.perf_counter()
        # Batch 0 has no predecessor, and _batch_end is still at its epoch-start zero.
        if self._batch_end > 0.0:
            self._data_times.append(now - self._batch_end)
        self._step_start = now

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        """Charge the time since the batch started to the step itself."""
        now = time.perf_counter()
        self._step_times.append(now - self._step_start)
        self._epoch_samples += batch[0].size(0)
        self._batch_end = now

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the epoch's throughput, unless the epoch ran no training step."""
        if not self._step_times:
            return
        metrics = self._epoch_metrics()
        for name, value in metrics.items():
            pl_module.log(name, value, on_step=False, on_epoch=True, sync_dist=False)
        summary = "  ".join(
            f"{name.removeprefix('perf/')}: {value:.2f}" for name, value in metrics.items()
        )
        log.info(f"[epoch {trainer.current_epoch:>3d}] {summary}")

    def _epoch_metrics(self) -> dict[str, float]:
        """Reduce the per-step timings of the finished epoch to the logged metrics."""
        total_time = time.perf_counter() - self._epoch_start
        rate = 1.0 / total_time if total_time > 0 else float("nan")
        return {
            "perf/train_epoch_s": total_time,
            "perf/train_steps_per_sec": rate * len(self._step_times),
            "perf/train_samples_per_sec": rate * self._epoch_samples,
            "perf/avg_step_ms": _mean_ms(self._step_times),
            "perf/avg_data_ms": _mean_ms(self._data_times),
            "perf/gpu_util_pct": _gpu_utilization(),
            "perf/gpu_mem_alloc_gb": _allocated_gb(),
        }


def _mean_ms(durations: list[float]) -> float:
    """Mean of a list of seconds in milliseconds, NaN when the list is empty."""
    if not durations:
        return float("nan")
    return 1000 * sum(durations) / len(durations)


def _allocated_gb() -> float:
    """GPU memory the caching allocator holds for tensors, in GB."""
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
