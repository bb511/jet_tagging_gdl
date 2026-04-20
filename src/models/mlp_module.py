from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary
from torch.nn.functional import cross_entropy
import torch.nn as nn

from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from src.models.mlp import MLP


class MLPClassifier(LightningModule):
    """Lightning wrapper around the MLP for jet classification.

    Expects input tensors of shape ``(batch, nconst, nfeats)``. They are
    flattened to ``(batch, nconst * nfeats)`` before being fed to the MLP.

    Supported methods:
        - augmentation   : data is pre-augmented on disk; standard CE loss
        - symm_loss      : unaugmented data + symmetry regularisation loss
        - another_method : placeholder, uses standard CE loss
    """

    SUPPORTED_METHODS = {"augmentation", "symm_loss", "another_method"}

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
        group: str = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.group = group

        # Validate method
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Expected one of: {self.SUPPORTED_METHODS}."
            )
        self.method = method

        # Example input for torchinfo and RichModelSummary
        self.example_input_array = torch.randn(2, nconstituents, nbr_feats)

        # Build network
        in_dim = nbr_feats * nconstituents
        self.net = MLP(
            in_dim=in_dim,
            nodes=hidden_layers,
            out_dim=output_dim,
            batchnorm=batchnorm,
            activation=activation,
        )

        # Loss — symm_loss uses the combined criterion, everything else uses CE
        if self.method == "symm_loss":
            self.criterion = combine_with_symmetry_loss
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc    = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc      = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_acc     = Accuracy(task="multiclass", num_classes=output_dim)
        self.train_loss   = MeanMetric()
        self.val_loss     = MeanMetric()
        self.test_loss    = MeanMetric()
        self.val_acc_best = MaxMetric()

        # feature_names populated in setup() from the datamodule
        self.feature_names = None

        # ----------------------------------------------------------------
        # Compute FLOPs in __init__ — model is fully built, no trainer
        # needed. Stored in hparams so it is:
        #   • available from the very first hook (incl. sanity-check val)
        #   • persisted in the checkpoint
        #   • automatically logged to MLflow via log_hyperparameters
        # ----------------------------------------------------------------
        stats = torchinfo_summary(
            self,
            input_data=self.example_input_array,
            verbose=0,
        )
        mult_adds = float(stats.total_mult_adds)
        total_flops = 2.0 * mult_adds
        # Keep backwards compatibility (`flops`) but store explicit names too.
        self.hparams["flops_mult_adds"] = mult_adds
        self.hparams["flops_total"] = total_flops
        self.hparams["flops"] = total_flops

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = x.flatten(start_dim=1)   # (B, nconst, nfeats) → (B, nconst*nfeats)
        return self.net(x)

    # ------------------------------------------------------------------
    # Shared step — branched cleanly by method
    # ------------------------------------------------------------------
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch

        # Support both one-hot and class-index labels
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)

        if self.method == "symm_loss":
            # Build symmetry-augmented batch on the fly.
            # feature_names is required by build_symmetric_batch and is
            # populated from the datamodule in setup().
            if self.feature_names is None:
                raise RuntimeError(
                    "feature_names is None in model_step. "
                    "Ensure setup() has been called so the datamodule's "
                    "feature_names are available before training starts."
                )
            x_sym, _ = build_symmetric_batch(
                x,
                group="so2",
                feature_names=self.feature_names,
            )
            logits_sym = self.forward(x_sym)
            cls_loss = cross_entropy(logits, y)
            loss, _ = self.criterion(cls_loss, logits, logits_sym, weight=0.1)

        else:
            # augmentation / another_method — plain cross-entropy
            # (augmentation differences are handled at the data level)
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # ------------------------------------------------------------------
    # Setup — grab feature_names from datamodule once trainer is attached
    # ------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            self.feature_names = self.trainer.datamodule.feature_names
            if self.method == "symm_loss" and self.feature_names is None:
                raise RuntimeError(
                    "datamodule.feature_names is None after setup(). "
                    "Ensure HLS4MLDataModule.setup() populates feature_names "
                    "from norm_params.npz before model.setup() is called."
                )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def on_train_start(self) -> None:
        # Reset val metrics so sanity-check values don't bleed into epoch 0
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
        self.log("train/acc",  self.train_acc,  on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = self.train_loss.compute()
        train_acc  = self.train_acc.compute()
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
        self.log("val/acc",  self.val_acc,  on_step=False, on_epoch=True, prog_bar=True)

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
        self.log("test/acc",  self.test_acc,  on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor":   "val/acc",
                    "interval":  "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
