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
    activations = {
        "relu":    lambda: nn.ReLU(inplace=True),
        "tanh":    lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
    factory = activations.get(name)
    if factory is None:
        raise ValueError(
            f"Activation '{name}' not implemented. Add it to _get_act() in irc_deepsets_module.py."
        )
    return factory()


class DeepSetsInvariantIRC(LightningModule):
    """IRC-safe Deep Sets classifier for jet tagging.

    IRC safety is achieved by:
      - feeding only (eta, phi) — indices 1:3 of the feature vector — into phi,
        so the network is insensitive to collinear/soft-emission kinematics
      - weighting each particle's phi output by its pt fraction before aggregation

    The pt fraction is computed per-jet: z_i = pt_i / sum_j(pt_j).
    Phi takes (eta, phi) → output, then z_i * phi_out is summed/meaned/maxed.
    Rho maps the aggregated representation to class logits.

    Supported methods: augmentation, symm_loss, another_method  (same as DeepSetsInvariant).
    """

    SUPPORTED_METHODS = {"augmentation", "symm_loss", "another_method"}

    # phi always receives 2 features: (eta, phi)
    PHI_INPUT_DIM = 2

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

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Expected one of: {self.SUPPORTED_METHODS}."
            )
        self.method       = method
        self.group        = group
        self.activ_name   = activation
        self.aggregation  = aggregation
        self.mask_enabled = mask

        # Full input shape for torchinfo / RichModelSummary
        self.example_input_array = torch.randn(2, 32, nbr_feats)

        # phi: takes (eta, phi) → always 2-dimensional input
        phi_blocks: list[nn.Module] = []
        dims = [self.PHI_INPUT_DIM] + list(phi_layers)
        for i in range(len(dims) - 1):
            phi_blocks.append(nn.Linear(dims[i], dims[i + 1]))
            phi_blocks.append(_get_act(self.activ_name))
        self.phi = nn.Sequential(*phi_blocks)

        # rho: maps aggregated representation to logits
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

        self.feature_names = None
        self.norm_scales   = None

        stats = torchinfo_summary(self, input_data=self.example_input_array, verbose=0)
        mult_adds = float(stats.total_mult_adds)
        total_flops = 2.0 * mult_adds
        self.hparams["flops_mult_adds"] = mult_adds
        self.hparams["flops_total"]     = total_flops
        self.hparams["flops"]           = total_flops

    # ------------------------------------------------------------------
    # Forward — IRC-safe: phi sees only (eta, phi), weighted by pt fraction
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        # pt fraction: (B, N)
        pt = x[:, :, 0]
        pt_sum = pt.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pt_frac = (pt / pt_sum).unsqueeze(-1)   # (B, N, 1)

        eta_phi = x[:, :, 1:3]                  # (B, N, 2)
        phi_out = self.phi(eta_phi)              # (B, N, phi_layers[-1])

        # IRC-safe weighting
        weighted = pt_frac * phi_out             # (B, N, phi_layers[-1])

        if self.mask_enabled:
            mask = (pt.abs() > 0).float().unsqueeze(-1)
            weighted = weighted * mask

        if self.aggregation == "sum":
            pooled = weighted.sum(dim=1)
        elif self.aggregation == "mean":
            pooled = weighted.mean(dim=1)
        elif self.aggregation == "max":
            pooled = weighted.max(dim=1).values
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
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        logits = self.forward(x)

        if self.method == "symm_loss":
            if self.feature_names is None:
                raise RuntimeError(
                    "feature_names is None — symm_loss requires feature_names. "
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
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

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
