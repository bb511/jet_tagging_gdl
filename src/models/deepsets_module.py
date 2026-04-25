from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary
from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy
import torch.nn as nn


def _get_act(name: str) -> nn.Module:
    """Return a *new* activation module from its name."""
    activations = {
        "relu":    lambda: nn.ReLU(inplace=True),
        "tanh":    lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
    factory = activations.get(name)
    if factory is None:
        raise ValueError(
            f"Activation '{name}' not implemented. Add it to _get_act() in deepsets_module.py."
        )
    return factory()


class DeepSetsInvariant(LightningModule):
    """Deep Sets permutation-invariant network for jet classification.

    Architecture (Zaheer et al.):
        phi : per-particle MLP → aggregation (sum/mean/max) → rho : post-aggregation MLP

    Supported methods:
        - augmentation   : data pre-augmented on disk; standard CE loss
        - symm_loss      : unaugmented data + symmetry regularisation loss
        - another_method : placeholder, uses standard CE loss
    """

    SUPPORTED_METHODS = {"augmentation", "symm_loss", "another_method"}

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
        compile: bool = False,
        group: str = "so2",
        symm_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Validate method
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Expected one of: {self.SUPPORTED_METHODS}."
            )
        self.method       = method
        self.group        = group
        self.activ_name   = activation
        self.aggregation  = aggregation
        self.mask_enabled = mask

        # Example input for torchinfo and RichModelSummary
        self.example_input_array = torch.randn(2, 32, nbr_feats)

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

        # Loss
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

        # feature_names and norm_scales populated in setup() from the datamodule
        self.feature_names = None
        self.norm_scales = None

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
        phi_out = self.phi(x)

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
            # Use the *selected* feature names (matching tensor dimensions), not
            # the raw full list from the H5 file which has j1_-prefixed names at
            # indices that no longer correspond to the feature-selected tensor.
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
        print(
            f"[Epoch {self.current_epoch:>3d}] "
            f"Train | loss: {train_loss:.4f}  acc: {train_acc:.4f}",
            flush=True,
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
            f"[Epoch {self.current_epoch:>3d}]   Val | loss: {val_loss:.4f}  acc: {acc:.4f}  best: {self.val_acc_best.compute():.4f}",
            flush=True,
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
