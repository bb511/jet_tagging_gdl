from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary
from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy
from src.models.mlp import MLP
from torch import nn



class MLPClassifier(LightningModule):
    """Lightning wrapper around the MLP for jet classification.

    Expects input tensors of shape ``(batch, nconst, nfeats)``.  They are
    flattened to ``(batch, nconst * nfeats)`` before being fed to the MLP.
    """

    def __init__(
        self,
        nbr_feats: int,
        nconstituents: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: str = "relu",
        method: str = "augmentation",
        batchnorm: bool = False,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        compile: bool = False,
        group: str = "so2", 
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.example_input_array = torch.randn(2, nconstituents, nbr_feats)

        in_dim = nbr_feats * nconstituents
        self.net = MLP(
            in_dim=in_dim,
            nodes=hidden_layers,
            out_dim=output_dim,
            batchnorm=batchnorm,
            activation=activation,
        )

        # Add method for symmetry loss
        self.method = method
        self.group = group

        # Cache feature_names for Lorentz group
        self.feature_names = None

        # loss function
        if self.method == "augmentation":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.method == "symm_loss":
            self.criterion = combine_with_symmetry_loss


        # metric objects
        self.train_acc = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_acc = Accuracy(task="multiclass", num_classes=output_dim)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

        self.group = group

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, nconst, nfeats) -> (batch, nconst * nfeats)
        x = x.to(torch.float32)
        x = x.flatten(start_dim=1)
        return self.net(x)

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
                if self.feature_names is None:
                    # Try to get from datamodule if not already cached
                    if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "feature_names"):
                        self.feature_names = self.trainer.datamodule.feature_names
                    else:
                        raise ValueError("feature_names must be provided for Lorentz transforms and could not be found.")
                x_sym, _ = build_symmetric_batch(x, group=self.group, feature_names=self.feature_names)
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
        # Ensure FLOPs is set for logging
        if "flops" not in self.hparams:
            # Move example input to the model's device to avoid device mismatch
            example_input = self.example_input_array.to(self.device)
            stats = torchinfo_summary(
                self,
                input_data=example_input,
                verbose=0,
            )
            self.flops = float(stats.total_mult_adds)
            self.hparams["flops"] = self.flops

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
        self.log("model/flops", self.hparams["flops"])
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
                feature_names = None
                if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "feature_names"):
                    feature_names = self.trainer.datamodule.feature_names
                if feature_names is None:
                    raise ValueError("feature_names must be provided for Lorentz transforms and could not be found in datamodule.")
                x_sym, _ = build_symmetric_batch(x, group=self.group, feature_names=feature_names)
            else:
                x_sym, _ = build_symmetric_batch(x, group=self.group)
            cls_loss = cross_entropy(logits, y)
            logits_sym = self.forward(x_sym)
            loss = self.criterion(cls_loss, logits, logits_sym, weight=0.1)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y

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
