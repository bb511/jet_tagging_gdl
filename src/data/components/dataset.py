# In-memory dataset that hands the trainer whole batches of jets.

from collections.abc import Callable

import torch
from torch.utils.data import IterableDataset


def _worker_slice() -> tuple[int, int]:
    """This worker's index and the worker count, (0, 1) in the main process."""
    info = torch.utils.data.get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


class HLS4MLDataset(IterableDataset):
    """Yield pre-batched jets from tensors held in memory.

    The whole split fits in RAM, so batching and shuffling are done here rather than by a
    sampler: the DataLoader is configured with ``batch_size=1`` and each of its items is one
    batch.

    :param x: the constituents, shape (jets, constituents, features).
    :param y: the one-hot flavour labels, shape (jets, classes).
    :param batch_size: a trailing partial batch is dropped, so ``len(self)`` is exact and
        OneCycleLR can be given a step count.
    :param shuffler: seeded generator for the per-epoch permutation, or None to keep the stored
        order, which is what the evaluation splits want.
    :param max_batches: cap on the batches each worker yields, for debug runs.
    :param transform: online augmentation, from a batch to ``(x, keep_mask)``. The mask marks the
        jets that survive an acceptance cut, and the labels follow it.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffler: torch.Generator | None = None,
        max_batches: int | None = None,
        transform: Callable | None = None,
    ):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples.")
        self.X = x
        self.y = y
        self.batch_size = batch_size
        self.shuffler = shuffler
        self.max_batches = max_batches
        self.transform = transform
        self.n = x.shape[0]
        self.num_batches = self.n // batch_size

    def __len__(self):
        if self.max_batches is None:
            return self.num_batches
        return min(self.num_batches, self.max_batches)

    def __iter__(self):
        order = self._order()
        worker_id, num_workers = _worker_slice()
        emitted = 0
        # Workers stride through the batches, so each batch is emitted exactly once.
        for i in range(worker_id, self.num_batches, num_workers):
            if self.max_batches is not None and emitted >= self.max_batches:
                break
            yield self._batch(order[i * self.batch_size : (i + 1) * self.batch_size])
            emitted += 1

    def _order(self) -> torch.Tensor:
        """The jet order of one epoch: a fresh permutation, or the order the tensors are in."""
        if self.shuffler is None:
            return torch.arange(self.n)
        return torch.randperm(self.n, generator=self.shuffler)

    def _batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One batch of jets and labels, transformed when the dataset carries a transform."""
        x, y = self.X[idx], self.y[idx]
        if self.transform is None:
            return x, y
        x, keep_mask = self.transform(x)
        return (x, y) if keep_mask is None else (x, y[keep_mask])
