from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
import torch.nn as nn



from torch.nn.functional import cross_entropy
from torchinfo import summary as torchinfo_summary


def _get_act(name: str) -> nn.Module:
    """Return a *new* activation module from its name."""
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
    factory = activations.get(name)
    if factory is None:
        raise ValueError(
            f"Activation '{name}' not implemented! Add it to _get_act() in deepsets_module.py."
        )
    return factory()


class DeepSetsInvariant(LightningModule):
    """Deep Sets permutation-invariant network for jet classification.

    Architecture (Zaheer et al.):
        phi : per-particle MLP  ->  aggregation (sum/mean/max)  ->  rho : post-aggregation MLP
    """

    def __init__(
        self,
        nbr_feats: int,
        phi_layers: list,
        rho_layers: list,
        activation: str,
        method: str,
        aggregation: str,
        dropout: float,
        output_dim: int,
        mask: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        compile: bool,
        group: str,
    ) -> None:
        super().__init__()

        # stores all init args in self.hparams and in the checkpoint
        self.save_hyperparameters(logger=False)

        self.activ_name = activation

        # Symmetry method
        self.method = method
        self.group = group

        self.aggregation = aggregation
        self.mask_enabled = mask

        # --- phi network (per-particle) ---
        phi_blocks: list[nn.Module] = []
        dims = [nbr_feats] + list(phi_layers)
        for i in range(len(dims) - 1):
            phi_blocks.append(nn.Linear(dims[i], dims[i + 1]))
            phi_blocks.append(_get_act(self.activ_name))
        self.phi = nn.Sequential(*phi_blocks)

        # --- rho network (post-aggregation) ---
        rho_blocks: list[nn.Module] = []
        dims = [phi_layers[-1]] + list(rho_layers) + [output_dim]
        for i in range(len(dims) - 1):
            if 0.0 < dropout < 1.0:
                rho_blocks.append(nn.Dropout(p=dropout))
            rho_blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                rho_blocks.append(_get_act(self.activ_name))
        self.rho = nn.Sequential(*rho_blocks)

        # loss function
        if self.method == "augmentation":
            self.criterion = torch.nn.CrossEntropyLoss()
        
        elif self.method == "symm_loss":
            self.criterion = combine_with_symmetry_loss

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_acc = Accuracy(task="multiclass", num_classes=output_dim)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        phi_out = self.phi(x)

        # zero-out phi outputs for padded (zero) constituents
        if self.mask_enabled:
            mask = (x.abs().sum(dim=2) > 0).float().unsqueeze(-1)
            phi_out = phi_out * mask

        if self.aggregation == "sum":
            pooled = phi_out.sum(dim=1)
        elif self.aggregation == "mean":
            pooled = phi_out.mean(dim=1)
        elif self.aggregation == "max":
            pooled = phi_out.max(dim=1).values
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

        return self.rho(pooled)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        # convert one-hot labels to class indices if needed
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        logits = self.forward(x)
        if self.method == "augmentation":
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y

        else:
            if self.group.lower() == "lorentz":
                feature_names = self.trainer.datamodule.feature_names if hasattr(self.trainer, "datamodule") else None
                x_sym, _ = build_symmetric_batch(x, group=self.group, feature_names=feature_names)
            else:
                x_sym, _ = build_symmetric_batch(x, group=self.group)
            cls_loss = cross_entropy(logits, y)
            logits_sym = self.forward(x_sym)
            loss = self.criterion(cls_loss, logits, logits_sym, weight=0.1)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()
        print(
            f"Epoch {self.current_epoch} ended. "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        val_loss = self.val_loss.compute()
        print(
            f"Epoch {self.current_epoch} ended. "
            f"Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}, "
            f"Val Acc Best: {self.val_acc_best.compute():.4f}"
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Setup & optimizers
    # ------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/acc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
