import torch
from torch.utils.data import IterableDataset
from typing import Callable, Optional


class HLS4MLDataset(IterableDataset):
    """In-memory jet dataset that yields pre-batched tensors directly.

    Shuffling is handled internally via a torch.Generator so the DataLoader
    can be configured with batch_size=1 and num_workers=0.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffler: Optional[torch.Generator] = None,
        max_batches: Optional[int] = None,
        transform: Optional[Callable] = None,
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
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        if self.shuffler is not None:
            perm = torch.randperm(self.n, generator=self.shuffler)
        else:
            perm = torch.arange(self.n)

        nb = 0
        for i in range(worker_id, self.num_batches, num_workers):
            if self.max_batches is not None and nb >= self.max_batches:
                break
            idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            x = self.X[idx]
            y = self.y[idx]
            if self.transform is not None:
                x, keep_mask = self.transform(x)
                if keep_mask is not None:
                    y = y[keep_mask]
            yield x, y
            nb += 1
