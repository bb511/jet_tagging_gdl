import logging
import time

import torch
import lightning as L

log = logging.getLogger(__name__)


class ThroughputCallback(L.Callback):
    """Logs training throughput and GPU utilisation at the end of each epoch.

    Metrics logged (via pl_module.log, so they appear in MLflow and the progress bar):
        perf/train_epoch_s         — total wall time for the epoch
        perf/train_steps_per_sec   — batches processed per second
        perf/train_samples_per_sec — samples processed per second
        perf/avg_step_ms           — mean wall time per training step (ms)
                                     (GPU forward + loss; excludes data loading)
        perf/avg_data_ms           — mean inter-batch wait (ms)
                                     ≈ DataLoader fetch + collate/augmentation time
        perf/gpu_util_pct          — torch.cuda.utilization() snapshot at epoch end
        perf/gpu_mem_alloc_gb      — GPU memory allocated at epoch end (GB)

    For online-augmentation runs, perf/avg_data_ms captures the collate-fn
    transform time (which runs inside worker processes and cannot be timed
    directly from the training step).  For symm_loss runs it reflects pure
    I/O, making the two directly comparable against perf/transform_ms.
    """

    def __init__(self) -> None:
        super().__init__()
        self._step_start: float = 0.0
        self._batch_end: float = 0.0
        self._epoch_start: float = 0.0
        self._step_times: list[float] = []
        self._data_times: list[float] = []
        self._epoch_samples: int = 0

    # ------------------------------------------------------------------
    # Per-step hooks
    # ------------------------------------------------------------------
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        now = time.perf_counter()
        # Inter-batch gap = time the main process spent waiting for this batch
        # (DataLoader fetch + collate + any augmentation in the collate_fn).
        # Skip batch 0: _batch_end is 0.0 (no previous batch end recorded yet).
        if self._batch_end > 0.0:
            self._data_times.append(now - self._batch_end)
        self._step_start = now

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        now = time.perf_counter()
        self._step_times.append(now - self._step_start)
        self._epoch_samples += batch[0].size(0)
        self._batch_end = now

    # ------------------------------------------------------------------
    # Epoch-level hooks
    # ------------------------------------------------------------------
    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._step_times = []
        self._data_times = []
        self._epoch_samples = 0
        self._batch_end = 0.0
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not self._step_times:
            return

        n_steps = len(self._step_times)
        total_time = time.perf_counter() - self._epoch_start
        avg_step_ms = (sum(self._step_times) / n_steps) * 1000
        avg_data_ms = (
            (sum(self._data_times) / len(self._data_times)) * 1000
            if self._data_times else float("nan")
        )
        steps_per_sec = n_steps / total_time if total_time > 0 else float("nan")
        samples_per_sec = self._epoch_samples / total_time if total_time > 0 else float("nan")

        try:
            gpu_util = torch.cuda.utilization() if torch.cuda.is_available() else 0.0
        except Exception:
            gpu_util = float("nan")
        gpu_mem_gb = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        pl_module.log("perf/train_epoch_s",          total_time,      on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("perf/train_steps_per_sec",   steps_per_sec,   on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("perf/train_samples_per_sec", samples_per_sec, on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("perf/avg_step_ms",           avg_step_ms,     on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("perf/avg_data_ms",           avg_data_ms,     on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("perf/gpu_util_pct",          gpu_util,        on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("perf/gpu_mem_alloc_gb",      gpu_mem_gb,      on_step=False, on_epoch=True, sync_dist=False)

        log.info(
            f"[Epoch {trainer.current_epoch}] "
            f"epoch_s: {total_time:.1f}s  "
            f"steps/sec: {steps_per_sec:.2f}  "
            f"samples/sec: {samples_per_sec:.0f}  "
            f"avg_step_ms: {avg_step_ms:.1f}  "
            f"avg_data_ms: {avg_data_ms:.1f}  "
            f"gpu_util: {gpu_util:.0f}%  "
            f"gpu_mem: {gpu_mem_gb:.2f} GB"
        )
