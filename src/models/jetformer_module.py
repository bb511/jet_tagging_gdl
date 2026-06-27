import logging
from typing import Any, Dict, Tuple, Optional

log = logging.getLogger(__name__)

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary
from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy
import torch.nn as nn
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models.components.jetformer_layers import SeqNorm, JetFormerBlock


class JetFormer(LightningModule):
    """Custom-attention replica of the reference JetFormer architecture (arXiv:2601.17215).

    Unlike src.models.transformer_module.Transformer (which relies on
    nn.TransformerEncoderLayer), this builds the attention/FFN blocks from
    scratch to match the reference parameter count and computation graph,
    including its lack of a padding mask.

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
        embed_dim: int,
        num_heads: int,
        num_transformers: int,
        output_size: int,
        method: str = "augmentation",
        group: str = "so2",
        symm_loss_weight: float = 0.1,
        normalization: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Expected one of: {self.SUPPORTED_METHODS}."
            )
        self.method    = method
        self.group     = group
        self.nbr_const = nbr_const
        self.nbr_feat  = nbr_feat

        self.example_input_array = torch.randn(2, nbr_const, nbr_feat)

        # Architecture
        num_tokens = nbr_const + 1  # +1 for the CLS token
        self.embedding  = nn.Linear(nbr_feat, embed_dim)
        self.input_norm = SeqNorm(embed_dim, normalization)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks     = nn.ModuleList(
            [
                JetFormerBlock(embed_dim, num_heads, num_tokens, dropout, activation, normalization)
                for _ in range(num_transformers)
            ]
        )
        self.classifier = nn.Linear(embed_dim, output_size)

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

        # feature_names, model_feature_indices and norm_scales populated in
        # setup() from the datamodule
        self.feature_names = None
        self.model_feature_indices = None
        self.norm_scales = None

        stats = torchinfo_summary(
            self,
            input_data=self.example_input_array,
            verbose=0,
        )
        mult_adds = float(stats.total_mult_adds)
        total_flops = 2.0 * mult_adds
        self.hparams["flops_mult_adds"] = mult_adds
        self.hparams["flops_total"] = total_flops
        self.hparams["flops"] = total_flops

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        # symm_loss callers (model_step, but also callbacks that call the
        # module directly, e.g. plotting/invariance) may pass the full
        # on-disk feature set rather than the model's reduced input width —
        # reduce here so every caller can just hand over whatever the
        # dataloader yields.
        if self.model_feature_indices is not None and x.shape[-1] != self.nbr_feat:
            x = x[:, :, self.model_feature_indices]

        B = x.size(0)

        x = self.embedding(x)    # (B, nbr_const, embed_dim)
        x = self.input_norm(x)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)  # (B, 1+nbr_const, embed_dim)

        for block in self.blocks:
            x = block(x)

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

        if self.method == "symm_loss":
            if self.feature_names is None:
                raise RuntimeError(
                    "feature_names is None — symm_loss requires feature_names for all groups. "
                    "Ensure setup() has been called so the datamodule populates it."
                )
            # x arrives with the full on-disk feature set here (the datamodule
            # defers reduction for symm_loss) so px/py/pz are available to
            # compute the transform. forward() reduces to the model's own
            # input features internally — same separation as the online
            # augmentation transform uses.
            x_sym, _ = build_symmetric_batch(
                x,
                group=self.group,
                feature_names=self.feature_names,
                norm_scales=self.norm_scales,
            )

            logits     = self.forward(x)
            logits_sym = self.forward(x_sym)
            cls_loss   = cross_entropy(logits, y)
            loss, _    = self.criterion(
                cls_loss, logits, logits_sym, weight=self.hparams.symm_loss_weight
            )

        else:
            logits = self.forward(x)
            loss   = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # ------------------------------------------------------------------
    # Setup — grab feature_names from datamodule once trainer is attached
    # ------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            # Full on-disk schema, regardless of method — symm_loss's
            # build_symmetric_batch needs px/py/pz, and so does any callback
            # (plotting/invariance) that hands forward() a wider-than-nbr_feat
            # tensor (e.g. when the datamodule keeps full test features for
            # InvarianceMetricCallback). forward() only reduces when the input
            # width actually differs from nbr_feat, so normal train/val/test
            # steps — which already receive pre-reduced input — are unaffected.
            self.feature_names = list(dm.feature_names)
            if not self.feature_names:
                raise RuntimeError(
                    "datamodule.feature_names is empty after setup(). "
                    "Ensure HLS4MLDataModule is configured with a features list."
                )
            self.model_feature_indices = self._compute_model_feature_indices(
                dm.features, self.feature_names
            )
            raw_params = getattr(dm, "norm_params", None)
            self.norm_scales = (
                {name: float(scale) for name, (_, scale) in raw_params.items()}
                if raw_params else None
            )

    @staticmethod
    def _compute_model_feature_indices(model_features, full_feature_names) -> list:
        indices = []
        for feat in model_features:
            if feat in full_feature_names:
                indices.append(full_feature_names.index(feat))
            elif f"j1_{feat}" in full_feature_names:
                indices.append(full_feature_names.index(f"j1_{feat}"))
            else:
                raise ValueError(
                    f"Requested feature '{feat}' not found in saved names: {full_feature_names}"
                )
        return indices

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
