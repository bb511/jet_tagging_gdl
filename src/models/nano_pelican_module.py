from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import logging

from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchinfo import summary as torchinfo_summary

from src.utils.custom_loss import combine_with_symmetry_loss, build_symmetric_batch
from torch.nn.functional import cross_entropy

from ..utils.lorentz_metric import dot4
from ..utils.perm_equiv_models import Net2to2, Eq2to0, MessageNet


# ---------------------------------------------------------------------------
# PELICANNano — bare nn.Module (Lorentz + permutation invariant)
# ---------------------------------------------------------------------------

class PELICANNano(nn.Module):
    """Permutation Invariant, Lorentz Invariant Aggregator Network (nano variant).

    Operates on pairwise Lorentz dot products of particle 4-momenta.
    Input data dict keys: 'Pmu' (B, N, 4), 'particle_mask' (B, N), 'edge_mask' (B, N, N).
    """

    def __init__(
        self,
        n_hidden: int,
        output_dim: int = 5,
        activate_agg: bool = False,
        activate_lin: bool = True,
        activation: str = "leakyrelu",
        config: str = "s",
        config_out: str = "s",
        average_nobj: int = 49,
        factorize: bool = False,
        masked: bool = True,
        activate_agg_out: bool = True,
        activate_lin_out: bool = False,
        dropout: bool = True,
        drop_rate: float = 0.05,
        drop_rate_out: float = 0.05,
        batchnorm=None,
    ):
        super().__init__()

        logging.info("Initializing PELICANNano")

        self.n_hidden = n_hidden
        self.output_dim = output_dim
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.config = config
        self.config_out = config_out
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        # Sequence of permutation-equivariant 2→2 blocks
        self.net2to2 = Net2to2(
            [1, n_hidden], [[]],
            activate_agg=activate_agg, activate_lin=activate_lin,
            activation=activation, dropout=dropout, drop_rate=drop_rate,
            batchnorm=batchnorm, config=config, average_nobj=average_nobj,
            factorize=factorize, masked=masked,
        )

        # Final 2→0 block: message layer + equivariant aggregation
        self.msg_2to0 = MessageNet(
            [n_hidden], activation=activation, batchnorm=batchnorm,
        )
        self.agg_2to0 = Eq2to0(
            n_hidden, output_dim,
            activate_agg=activate_agg_out, activate_lin=activate_lin_out,
            activation=activation, config=config_out, factorize=False,
            average_nobj=average_nobj,
        )

        logging.info("_________________________")
        for n, p in self.named_parameters():
            logging.info(f'{"Parameter: " + n:<80} {p.shape}')
        logging.info(
            "Model initialized. Parameters: {}".format(
                sum(p.nelement() for p in self.parameters())
            )
        )
        logging.info("_________________________")

    def prepare_input(self, data: dict):
        # Use parameter device so Lightning GPU placement is respected.
        device = next(self.parameters()).device
        particle_ps = data["Pmu"].to(device, torch.float32)
        particle_mask = data["particle_mask"].to(device, torch.bool)
        edge_mask = data["edge_mask"].to(device, torch.bool)
        return particle_mask, edge_mask, particle_ps

    def forward(self, data: dict, covariance_test: bool = False):
        particle_mask, edge_mask, event_momenta = self.prepare_input(data)

        dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2))
        inputs = dot_products.unsqueeze(-1)
        nobj = particle_mask.sum(-1, keepdim=True)

        act1 = self.net2to2(inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj)

        act2 = self.msg_2to0(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)
        act3 = self.agg_2to0(act2, nobj=nobj)  # (B, output_dim)

        if self.dropout:
            act3 = self.dropout_layer_out(act3)
        prediction = act3  # (B, output_dim) — multi-class logits

        if torch.isnan(prediction).any():
            logging.info(f"inputs NaN: {torch.isnan(inputs).any()}")
            logging.info(f"act1   NaN: {torch.isnan(act1).any()}")
            logging.info(f"act2   NaN: {torch.isnan(act2).any()}")
        assert not torch.isnan(prediction).any(), "NaN in PELICAN output."

        if covariance_test:
            return {"predict": prediction, "inputs": inputs, "act1": act1, "act2": act2, "act3": act3}
        return {"predict": prediction}


# ---------------------------------------------------------------------------
# PELICANNanoModule — LightningModule wrapper for the HLS4ML workflow
# ---------------------------------------------------------------------------

class PELICANNanoModule(LightningModule):
    """Lightning wrapper for PELICANNano compatible with the HLS4ML datamodule.

    Input: (B, N, F) tensor (same as DeepSets / IRC DeepSets).
    Internally reconstructs Lorentz 4-momenta [E, px, py, pz] from px/py/pz
    columns, computes pairwise dot products, and runs PELICANNano.

    Note: the datamodule normalises px/py/pz by a per-feature scale, so the
    resulting dot products are proportional to (but not equal to) physical
    Lorentz invariants.  Lorentz-invariance of the network representation is
    preserved; only the absolute scale of the dot products is different from
    raw-GeV inputs.
    """

    SUPPORTED_METHODS = {"augmentation", "symm_loss", "equivariance"}

    def __init__(
        self,
        n_hidden: int,
        output_dim: int,
        nbr_feats: int,
        nconstituents: int,
        method: str,
        group: str = "so2",
        symm_loss_weight: float = 0.1,
        # PELICANNano architecture kwargs
        activate_agg: bool = False,
        activate_lin: bool = True,
        activation: str = "leakyrelu",
        config: str = "s",
        config_out: str = "s",
        average_nobj: int = 8,
        factorize: bool = False,
        masked: bool = True,
        activate_agg_out: bool = True,
        activate_lin_out: bool = False,
        dropout: bool = True,
        drop_rate: float = 0.05,
        drop_rate_out: float = 0.05,
        batchnorm=None,
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

        self.net = PELICANNano(
            n_hidden=n_hidden,
            output_dim=output_dim,
            activate_agg=activate_agg,
            activate_lin=activate_lin,
            activation=activation,
            config=config,
            config_out=config_out,
            average_nobj=average_nobj,
            factorize=factorize,
            masked=masked,
            activate_agg_out=activate_agg_out,
            activate_lin_out=activate_lin_out,
            dropout=dropout,
            drop_rate=drop_rate,
            drop_rate_out=drop_rate_out,
            batchnorm=batchnorm,
        )

        if self.method == "symm_loss":
            self.criterion = combine_with_symmetry_loss
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.train_acc    = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc      = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_acc     = Accuracy(task="multiclass", num_classes=output_dim)
        self.train_loss   = MeanMetric()
        self.val_loss     = MeanMetric()
        self.test_loss    = MeanMetric()
        self.val_acc_best = MaxMetric()

        self.feature_names = None
        self.norm_scales   = None

        # Column indices — defaults for ["px","py","pz","etarel","phirel"]
        self._px_col = 0
        self._py_col = 1
        self._pz_col = 2  # set to None if pz is not in features

        stats = torchinfo_summary(self, input_data=self.example_input_array, verbose=0)
        mult_adds = float(stats.total_mult_adds)
        total_flops = 2.0 * mult_adds
        self.hparams["flops_mult_adds"] = mult_adds
        self.hparams["flops_total"]     = total_flops
        self.hparams["flops"]           = total_flops

    # ------------------------------------------------------------------
    # Input conversion: (B, N, F) → Lorentz 4-momenta + masks
    # ------------------------------------------------------------------
    def _features_to_pmu(self, x: torch.Tensor) -> torch.Tensor:
        """Build (B, N, 4) 4-momentum tensor [E, px, py, pz] from feature columns."""
        px = x[:, :, self._px_col]
        py = x[:, :, self._py_col]
        if self._pz_col is not None:
            pz = x[:, :, self._pz_col]
            E  = torch.sqrt(px**2 + py**2 + pz**2 + 1e-8)  # massless
        else:
            pz = torch.zeros_like(px)
            E  = torch.sqrt(px**2 + py**2 + 1e-8)
        return torch.stack([E, px, py, pz], dim=-1)  # (B, N, 4)

    def _make_masks(self, pmu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Derive particle and edge masks from transverse momentum."""
        pt = torch.sqrt(pmu[:, :, 1]**2 + pmu[:, :, 2]**2)
        particle_mask = pt > 1e-6                                        # (B, N)
        edge_mask = particle_mask.unsqueeze(1) & particle_mask.unsqueeze(2)  # (B, N, N)
        return particle_mask, edge_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        pmu = self._features_to_pmu(x)
        particle_mask, edge_mask = self._make_masks(pmu)
        data = {"Pmu": pmu, "particle_mask": particle_mask, "edge_mask": edge_mask}
        return self.net(data)["predict"]  # (B, output_dim)

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

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            self.feature_names = list(dm.features)
            raw_params = getattr(dm, "norm_params", None)
            self.norm_scales = (
                {name: float(scale) for name, (_, scale) in raw_params.items()}
                if raw_params else None
            )
            self._resolve_col_indices(self.feature_names)

    def _resolve_col_indices(self, names: list) -> None:
        def _find(candidates):
            for n in candidates:
                if n in names:
                    return names.index(n)
            return None

        self._px_col = _find(["px", "j1_px"])
        self._py_col = _find(["py", "j1_py"])
        self._pz_col = _find(["pz", "j1_pz"])

        if self._px_col is None or self._py_col is None:
            raise RuntimeError(
                f"PELICANNanoModule: px/py columns not found in features {names}. "
                "PELICAN requires px and py to reconstruct 4-momenta."
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
        print(
            f"[Epoch {self.current_epoch:>3d}] "
            f"Train | loss: {train_loss:.4f}  acc: {train_acc:.4f}",
            flush=True,
        )

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
        print(
            f"[Epoch {self.current_epoch:>3d}]   Val | loss: {val_loss:.4f}  acc: {acc:.4f}"
            f"  best: {self.val_acc_best.compute():.4f}",
            flush=True,
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
