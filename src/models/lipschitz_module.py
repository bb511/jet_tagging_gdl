import logging
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary

import deel.torchlip as torchlip

from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy
from pytorch_lightning.utilities.memory import garbage_collection_cuda


# ---------------------------------------------------------------------------
# 1-Lipschitz activation factory
# ---------------------------------------------------------------------------

def _get_lip_act(name: str) -> nn.Module:
    """Return a 1-Lipschitz activation by name.

    groupsort2 / maxmin both require even-dimensional input.
    fullsort works for any dimension.
    """
    if name == "groupsort2":
        return torchlip.GroupSort2()
    if name == "maxmin":
        return torchlip.MaxMin()
    if name == "fullsort":
        return torchlip.FullSort()
    raise ValueError(
        f"Unknown Lipschitz activation '{name}'. "
        "Choose one of: groupsort2, maxmin, fullsort."
    )


# ---------------------------------------------------------------------------
# LipschitzMLP — core 1-Lipschitz MLP (nn.Module)
# ---------------------------------------------------------------------------

class LipschitzMLP(nn.Module):
    """1-Lipschitz MLP built from SpectralLinear layers and GroupSort2 activations.

    All singular values of each weight matrix are constrained to 1 via Bjorck
    orthonormalisation, so the network is globally 1-Lipschitz.

    Args:
        in_dim:        Flattened input dimension (nconstituents * nbr_feats).
        hidden_layers: List of hidden layer widths. Must all be even when using
                       groupsort2 or maxmin activations.
        output_dim:    Number of output classes.
        activation:    One of 'groupsort2', 'maxmin', 'fullsort'.
        k_coef_lip:    Global Lipschitz coefficient passed to torchlip.Sequential.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_layers: list,
        output_dim: int,
        activation: str = "groupsort2",
        k_coef_lip: float = 1.0,
    ):
        super().__init__()

        if activation in ("groupsort2", "maxmin"):
            for d in hidden_layers:
                if d % 2 != 0:
                    raise ValueError(
                        f"'{activation}' activation requires even hidden dimensions; got {d}. "
                        "Use fullsort or make all hidden_layers widths even."
                    )

        layers: list[nn.Module] = []
        dims = [in_dim] + list(hidden_layers)
        for i in range(len(dims) - 1):
            layers.append(torchlip.SpectralLinear(dims[i], dims[i + 1], k_coef_lip=1.0))
            layers.append(_get_lip_act(activation))
        layers.append(torchlip.SpectralLinear(dims[-1], output_dim, k_coef_lip=1.0))

        self.net = torchlip.Sequential(*layers, k_coef_lip=k_coef_lip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# LipschitzMLPClassifier — LightningModule wrapper for the HLS4ML workflow
# ---------------------------------------------------------------------------

class LipschitzMLPClassifier(LightningModule):
    """Lightning wrapper for LipschitzMLP compatible with the HLS4ML datamodule.

    Accepts (B, N, F) tensors, flattens to (B, N*F), and runs through a
    1-Lipschitz MLP built from SpectralLinear + GroupSort2 layers.

    Supported methods:
        - augmentation : data pre-augmented on disk; standard CE loss
        - symm_loss    : unaugmented data + symmetry regularisation loss
        - equivariance : no augmentation, standard CE loss (baseline)
    """

    SUPPORTED_METHODS = {"augmentation", "symm_loss", "equivariance"}

    def __init__(
        self,
        nbr_feats: int,
        nconstituents: int,
        hidden_layers: list,
        output_dim: int,
        activation: str = "groupsort2",
        k_coef_lip: float = 1.0,
        method: str = "augmentation",
        group: str = "so2",
        symm_loss_weight: float = 0.1,
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
        self.method = method
        self.group = group

        self.example_input_array = torch.randn(2, nconstituents, nbr_feats)

        in_dim = nbr_feats * nconstituents
        self.net = LipschitzMLP(
            in_dim=in_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
            k_coef_lip=k_coef_lip,
        )

        self.criterion = combine_with_symmetry_loss if method == "symm_loss" else nn.CrossEntropyLoss()

        self.train_acc    = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc      = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_acc     = Accuracy(task="multiclass", num_classes=output_dim)
        self.train_loss   = MeanMetric()
        self.val_loss     = MeanMetric()
        self.test_loss    = MeanMetric()
        self.val_acc_best = MaxMetric()

        self.feature_names = None
        self.norm_scales   = None

        stats = torchinfo_summary(self, input_data=self.example_input_array, verbose=0)
        mult_adds = float(stats.total_mult_adds)
        total_flops = 2.0 * mult_adds
        self.hparams["flops_mult_adds"] = mult_adds
        self.hparams["flops_total"]     = total_flops
        self.hparams["flops"]           = total_flops

        logging.info(
            f"LipschitzMLPClassifier: in_dim={in_dim}, hidden={hidden_layers}, "
            f"out={output_dim}, activation={activation}, k_coef_lip={k_coef_lip}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = x.flatten(start_dim=1)   # (B, N, F) → (B, N*F)
        return self.net(x)

    # ------------------------------------------------------------------
    # Shared step
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
                    "feature_names is None — symm_loss requires feature_names. "
                    "Ensure setup() has been called."
                )
            x_sym, _ = build_symmetric_batch(
                x, group=self.group,
                feature_names=self.feature_names,
                norm_scales=self.norm_scales,
            )
            logits_sym = self.forward(x_sym)
            cls_loss   = cross_entropy(logits, y)
            loss, _    = self.criterion(
                cls_loss, logits, logits_sym, weight=self.hparams.symm_loss_weight
            )
        else:
            loss = self.criterion(logits, y)

        return loss, torch.argmax(logits, dim=1), y

    # ------------------------------------------------------------------
    # Setup
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
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
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
            f"[Epoch {self.current_epoch:>3d}]   Val | loss: {val_loss:.4f}  acc: {acc:.4f}"
            f"  best: {self.val_acc_best.compute():.4f}"
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
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
