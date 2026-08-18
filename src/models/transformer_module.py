import logging
from typing import Any, Dict, Tuple, Optional

log = logging.getLogger(__name__)

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utils.flops import record_flops
from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy
import torch.nn as nn
from pytorch_lightning.utilities.memory import garbage_collection_cuda


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
        symm_loss_weight: float = 0.1,
        norm: Optional[str] = None,
        use_mask: bool = True,
        fill_masked: bool = True,
        bias: bool = True,
        fast_path: bool = False,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
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
        self.cls_token          = nn.Parameter(torch.zeros(1, 1, d_models))
        self.encoder_layer      = nn.TransformerEncoderLayer(
            d_model=d_models,
            nhead=nhead,
            dim_feedforward=dim_feedforward if dim_feedforward is not None else 2048,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers, enable_nested_tensor=fast_path
        )
        self.classifier = nn.Linear(d_models, output_size)

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

        # feature_names and norm_scales populated in setup() from the datamodule
        self.feature_names = None
        self.norm_scales = None

        # ----------------------------------------------------------------
        # Compute FLOPs in __init__ — model fully built, no trainer needed.
        # Stored in hparams: available from the very first hook (including
        # sanity-check validation), persisted in checkpoints, and logged
        # to MLflow automatically via log_hyperparameters.
        # ----------------------------------------------------------------
        # Per-jet forward FLOPs (scales with the constituent count; see
        # src/utils/flops.py for the counting convention).
        record_flops(self, nbr_const, nbr_feat)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        B = x.size(0)

        mask = (x.sum(dim=-1) == 0) if self.use_mask else None  # (B, nbr_const)

        x = self.lifting_layer(x)  # (B, nbr_const, d_models)

        # Prepend CLS token; extend mask so CLS is never padding-masked
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_models)
        x   = torch.cat([cls, x], dim=1)              # (B, 1+nbr_const, d_models)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1) # (B, 1+nbr_const)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (B, 1+nbr_const, d_models)
        return self.classifier(x[:, 0, :])  # (B, output_size)

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
            if self.feature_names is None:
                raise RuntimeError(
                    "feature_names is None — symm_loss requires feature_names for all groups. "
                    "Ensure setup() has been called so the datamodule populates it."
                )
            x_sym, _ = build_symmetric_batch(
                x,
                group=self.group,
                feature_names=self.feature_names,
                norm_scales=self.norm_scales,
            )

            logits_sym = self.forward(x_sym)
            cls_loss   = cross_entropy(logits, y)
            loss, _    = self.criterion(
                cls_loss, logits, logits_sym, weight=self.hparams.symm_loss_weight
            )

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
            dm = self.trainer.datamodule
            self.feature_names = list(dm.features)
            if self.method == "symm_loss" and not self.feature_names:
                raise RuntimeError(
                    "datamodule.features is empty after setup(). "
                    "Ensure HLS4MLDataModule is configured with a features list."
                )
            raw_params = getattr(dm, "norm_params", None)
            self.norm_scales = (
                {name: float(scale) for name, (_, scale) in raw_params.items()}
                if raw_params else None
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
        log.info(
            f"[Epoch {self.current_epoch:>3d}] "
            f"Train | loss: {train_loss:.4f}  acc: {train_acc:.4f}"
        )
    
        garbage_collection_cuda()

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
        log.info(
            f"[Epoch {self.current_epoch:>3d}]   Val | loss: {val_loss:.4f}  acc: {acc:.4f}  best: {self.val_acc_best.compute():.4f}"
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
