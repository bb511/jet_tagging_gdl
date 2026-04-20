from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary
from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy
import torch.nn as nn


class Transformer(LightningModule):
    """Transformer encoder for jet classification.

    Based on "Attention is All You Need" (Vaswani et al., 2017).

    Supported methods:
        - augmentation   : data pre-augmented on disk; standard CE loss
        - symm_loss      : unaugmented data + symmetry regularisation loss
        - another_method : placeholder, uses standard CE loss
    """

    SUPPORTED_METHODS = {"augmentation", "symm_loss", "another_method"}

    def __init__(
        self,
        nbr_const: int,
        nbr_feat: int,
        lifting_dim: int,
        num_layers: int,
        d_models: int,
        nhead: int,
        output_size: int,
        method: str = "augmentation",
        group: str = "so2",
        norm: Optional[str] = None,
        use_mask: bool = True,
        fill_masked: bool = True,
        bias: bool = True,
        fast_path: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Validate method — must happen before any reference to self.method
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Expected one of: {self.SUPPORTED_METHODS}."
            )
        self.method      = method
        self.group       = group
        self.nbr_const   = nbr_const
        self.nbr_feat    = nbr_feat
        self.d_models    = d_models
        self.use_mask    = use_mask
        self.fill_masked = fill_masked

        # Example input for torchinfo and RichModelSummary
        self.example_input_array = torch.randn(2, nbr_const, nbr_feat)

        # Architecture
        self.lifting_layer      = nn.Linear(nbr_feat, d_models, bias=bias)
        self.encoder_layer      = nn.TransformerEncoderLayer(
            d_model=d_models, nhead=nhead, batch_first=True, bias=bias
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers, enable_nested_tensor=fast_path
        )
        self.fc = nn.Linear(d_models * nbr_const, output_size)

        # Loss
        if self.method == "symm_loss":
            self.criterion = combine_with_symmetry_loss
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc    = Accuracy(task="multiclass", num_classes=output_size)
        self.val_acc      = Accuracy(task="multiclass", num_classes=output_size)
        self.test_acc     = Accuracy(task="multiclass", num_classes=output_size)
        self.train_loss   = MeanMetric()
        self.val_loss     = MeanMetric()
        self.test_loss    = MeanMetric()
        self.val_acc_best = MaxMetric()

        # feature_names populated in setup() from the datamodule
        self.feature_names = None

        # ----------------------------------------------------------------
        # Compute FLOPs in __init__ — model fully built, no trainer needed.
        # Stored in hparams: available from the very first hook (including
        # sanity-check validation), persisted in checkpoints, and logged
        # to MLflow automatically via log_hyperparameters.
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

        mask = (x.sum(dim=-1) == 0) if self.use_mask else None  # (B, nbr_const)

        x = self.lifting_layer(x)                                          # (B, nbr_const, d_models)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)        # (B, nbr_const, d_models)

        # Zero out masked positions before flattening
        if self.use_mask and self.fill_masked and mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = x.view(x.size(0), -1)   # (B, nbr_const * d_models)
        return self.fc(x)            # (B, output_size)

    # ------------------------------------------------------------------
    # Shared step — branched cleanly by method
    # ------------------------------------------------------------------
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)

        if self.method == "symm_loss":
            # Lorentz boosts require feature_names; SO2/SO3 do not
            if self.group.lower() == "lorentz":
                if self.feature_names is None:
                    raise RuntimeError(
                        "feature_names is None — Lorentz symm_loss requires feature_names. "
                        "Ensure setup() has been called so the datamodule populates it."
                    )
                x_sym, _ = build_symmetric_batch(x, group=self.group, feature_names=self.feature_names)
            else:
                x_sym, _ = build_symmetric_batch(x, group=self.group)

            logits_sym = self.forward(x_sym)
            cls_loss   = cross_entropy(logits, y)
            loss, _    = self.criterion(cls_loss, logits, logits_sym, weight=0.1)

        else:
            # augmentation / another_method — plain cross-entropy
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # ------------------------------------------------------------------
    # Setup — grab feature_names from datamodule once trainer is attached
    # ------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            self.feature_names = self.trainer.datamodule.feature_names
            if (
                self.method == "symm_loss"
                and self.group.lower() == "lorentz"
                and self.feature_names is None
            ):
                raise RuntimeError(
                    "datamodule.feature_names is None after setup(). "
                    "Ensure HLS4MLDataModule.setup() populates feature_names from norm_params.npz."
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
            num_training_steps = self.trainer.estimated_stepping_batches
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_training_steps=num_training_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval":  "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
