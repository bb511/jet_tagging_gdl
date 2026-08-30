# Base LightningModule every jet-tagging architecture in this package subclasses.
"""Shared training logic for the jet-tagging architectures.

A subclass supplies only the network: it implements :meth:`JetTaggingModule.build_net` and, if its
inputs need a different parameterisation than the stored columns, overrides
:meth:`JetTaggingModule._prepare_inputs` and :meth:`JetTaggingModule._on_features_resolved`.
Everything else, the two training methods compared in this study included, lives here.

The two methods are ``augmentation``, where the datamodule transforms the training batches online
and the loss is plain cross-entropy, and ``symm_loss``, where the batches are left alone and a
symmetry-consistency term penalises the change in the logits under a transformation drawn per jet.
"""

import inspect
from collections.abc import Callable

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from src.algorithms.losses.symmetry import (
    SymmetryConsistencyLoss,
    build_symmetric_batch,
    validate_symmetry_group,
)
from src.algorithms.utils.flops import record_flops
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

SUPPORTED_METHODS = ("augmentation", "symm_loss")


def resolve_feature_indices(model_features: list[str], feature_names: list[str]) -> list[int]:
    """Column indices of *model_features* inside *feature_names*.

    Names match bare or with the record's ``j1_`` prefix, so ``px`` finds ``j1_px``.

    :raises ValueError: naming the first requested feature the data do not carry.
    """
    index_of = {name: idx for idx, name in enumerate(feature_names)}
    indices = []
    for feature in model_features:
        idx = index_of.get(feature, index_of.get(f"j1_{feature}"))
        if idx is None:
            raise ValueError(
                f"Feature {feature!r} is absent from the data columns {feature_names}. "
                f"data.features must contain {', '.join(model_features)}."
            )
        indices.append(idx)
    return indices


class JetTaggingModule(LightningModule):
    """Classify jets into five flavours, under either training method.

    Subclasses set :attr:`required_features` when the network reads named physical quantities
    rather than whatever columns ``data.features`` happens to list.
    """

    # Columns the network needs, in its own order; None means "whatever the data give".
    required_features: list[str] | None = None

    def __init__(
        self,
        name: str,
        nconstituents: int,
        nbr_feats: int,
        output_dim: int = 5,
        method: str = "augmentation",
        group: str = "none",
        symm_loss_weight: float = 0.1,
        symm_loss_metric: str = "l2",
        lorentz_eta_max: float = 2.0,
        optimizer: Callable | None = None,
        scheduler: dict | None = None,
        compile: bool = False,
        **net_kwargs,
    ) -> None:
        """
        :param name: short label for the results table and the MLflow run.
        :param nconstituents: constituents per jet, the second axis of a batch.
        :param nbr_feats: feature columns the network consumes, the third axis.
        :param output_dim: number of jet classes.
        :param method: ``augmentation`` or ``symm_loss``.
        :param group: symmetry group of the SCL, read only under ``symm_loss``.
        :param symm_loss_weight: weight of the symmetry term relative to cross-entropy.
        :param symm_loss_metric: distance between the two logit sets; see
            :func:`~src.algorithms.losses.symmetry.symmetry_distance`.
        :param lorentz_eta_max: half-width of the rapidity range of the SCL boosts.
        :param optimizer: Hydra ``_partial_`` optimiser, called with ``params``.
        :param scheduler: Lightning scheduler dict whose ``scheduler`` entry is a Hydra
            ``_partial_``; ``None`` trains at a fixed learning rate.
        :param compile: wrap the network in :func:`torch.compile` before fitting.
        :param net_kwargs: forwarded to :meth:`build_net`, and saved as hyperparameters.
        """
        super().__init__()
        # The optimiser and scheduler partials stay out of hparams so that a checkpoint
        # holds plain containers only and loads without add_safe_globals.
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler"])
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {list(SUPPORTED_METHODS)}, got {method!r}.")
        validate_symmetry_group(method, group)

        self._optimizer = optimizer
        self._scheduler = scheduler
        self.net = self.build_net(**net_kwargs)
        self.symm_loss = (
            SymmetryConsistencyLoss(symm_loss_weight, symm_loss_metric)
            if method == "symm_loss"
            else None
        )
        self._build_metrics(output_dim, method)

        # Filled by setup() from the datamodule, so None until the trainer attaches one.
        self.feature_names: list[str] | None = None
        self.model_feature_indices: list[int] | None = None
        self.norm_scales: dict[str, float] | None = None

        self.example_input_array = torch.randn(2, nconstituents, nbr_feats)
        # Counted here rather than in a callback: the network is complete, no trainer is
        # needed, and hparams reach the loggers and the checkpoint from the first hook.
        record_flops(self, nconstituents, nbr_feats)

    def _build_metrics(self, output_dim: int, method: str) -> None:
        """Register the metric objects; the SCL trackers exist only under ``symm_loss``."""
        self.train_acc = MulticlassAccuracy(num_classes=output_dim)
        self.val_acc = MulticlassAccuracy(num_classes=output_dim)
        self.test_acc = MulticlassAccuracy(num_classes=output_dim)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        self.train_symm_loss = MeanMetric() if method == "symm_loss" else None
        self.val_symm_loss = MeanMetric() if method == "symm_loss" else None

    def build_net(self, **net_kwargs) -> nn.Module:
        """Return the network this algorithm trains.

        Implemented by every subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement build_net().")

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Last transformation of a batch before the network sees it.

        The identity here; a subclass overrides it to de-normalise or reorder columns.
        """
        return x

    def _on_features_resolved(self) -> None:
        """Hook run at the end of :meth:`setup`, once the feature layout is known."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, constituents, features).

        Load-bearing width guard: training and validation batches arrive already at
        ``nbr_feats`` and pass straight through, whereas test batches (under
        ``keep_full_features_at_test``) and every SCL batch arrive at the record's full
        16-column width and are reduced here. Callbacks may therefore hand either width
        to the same module.
        """
        x = x.to(torch.float32)
        if self.model_feature_indices is not None and x.shape[-1] != self.hparams.nbr_feats:
            x = x[:, :, self.model_feature_indices]
        return self.net(self._prepare_inputs(x))

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run one forward pass and build the loss for it.

        :param batch: the pair ``(x, y)``, with ``x`` of shape (jets, constituents, features) and
            ``y`` a one-hot vector of shape (jets, classes).
        :return: a dict of ``loss``, ``cls_loss``, ``logits``, ``preds`` and ``targets``, plus
            ``symm_loss`` under ``method="symm_loss"``.
        """
        x, y = batch
        if y.ndim > 1 and y.shape[1] > 1:
            y = y.argmax(dim=1)
        logits = self(x)
        cls_loss = F.cross_entropy(logits, y)
        out = {
            "loss": cls_loss,
            "cls_loss": cls_loss,
            "logits": logits,
            "preds": logits.argmax(dim=1),
            "targets": y,
        }
        if self.symm_loss is not None:
            out["loss"], out["symm_loss"] = self._symmetry_branch(x, logits, cls_loss)
        return out

    def _symmetry_branch(
        self, x: torch.Tensor, logits: torch.Tensor, cls_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the SCL term, comparing the logits on *x* with those on a random T(x)."""
        if self.feature_names is None:
            raise RuntimeError(
                "feature_names is unset, so the columns of x cannot be named. Run under "
                "a trainer whose datamodule setup() has resolved the feature layout."
            )
        x_sym, keep_mask = build_symmetric_batch(
            x,
            self.hparams.group,
            self.feature_names,
            self.norm_scales,
            lorentz_eta_max=self.hparams.lorentz_eta_max,
        )
        if keep_mask is not None and not bool(keep_mask.any()):
            # The SO(3) acceptance filter rejected every jet, leaving no partner to
            # compare against; a mean over the empty batch would poison the loss with a
            # NaN, so this batch contributes cross-entropy alone.
            return cls_loss, cls_loss.new_zeros(())

        # SO(3) drops jets rotated out of acceptance, so keep only their partners' logits.
        kept = logits if keep_mask is None else logits[keep_mask]
        return self.symm_loss(cls_loss, kept, self(x_sym))

    def on_train_start(self) -> None:
        """Discard the sanity-check values so they never enter ``val/acc_best``."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        out = self.model_step(batch)
        self.train_loss(out["loss"])
        self.train_acc(out["preds"], out["targets"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        if self.train_symm_loss is not None:
            self.train_symm_loss(out["symm_loss"])
            self.log("train/symm_loss", self.train_symm_loss, on_step=False, on_epoch=True)
        return out["loss"]

    # The step dict is handed on to the callbacks: the plotting and invariance callbacks read
    # its logits rather than running the forward pass a second time on every batch.
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.model_step(batch)
        self.val_loss(out["loss"])
        self.val_acc(out["preds"], out["targets"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        if self.val_symm_loss is not None:
            self.val_symm_loss(out["symm_loss"])
            self.log("val/symm_loss", self.val_symm_loss, on_step=False, on_epoch=True)
        return out

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self.model_step(batch)
        self.test_loss(out["loss"])
        self.test_acc(out["preds"], out["targets"])
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return out

    def on_train_epoch_end(self) -> None:
        log.info(
            f"[epoch {self.current_epoch:>3d}] train | loss {self.train_loss.compute():.4f} "
            f"acc {self.train_acc.compute():.4f}"
        )

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        best = self.val_acc_best.compute()
        self.log("val/acc_best", best, sync_dist=True, prog_bar=True)
        log.info(
            f"[epoch {self.current_epoch:>3d}]   val | loss {self.val_loss.compute():.4f} "
            f"acc {acc:.4f} best {best:.4f}"
        )

    def setup(self, stage: str) -> None:
        """Resolve the feature layout from the datamodule, and compile the net if asked."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        # _trainer rather than trainer: the public property raises when the module stands
        # alone, which is how the unit tests exercise a bare model_step.
        datamodule = getattr(self._trainer, "datamodule", None)
        if datamodule is None:
            return
        self._resolve_features(datamodule)
        self.norm_scales = datamodule.norm_scales
        self._on_features_resolved()

    def _resolve_features(self, datamodule) -> None:
        """Record which columns of the incoming batches the network consumes.

        Under the SCL, and whenever the datamodule keeps all 16 record columns at test time,
        batches are wider than the network's input and the reduction indices are needed. Otherwise
        the batch already has the right width, and only a network with :attr:`required_features`
        reorders it.
        """
        model_features = self.required_features or list(datamodule.features)
        full_width = self.hparams.method == "symm_loss" or datamodule.keep_full_features_at_test
        if not full_width:
            self.feature_names = list(datamodule.features)
            self.model_feature_indices = (
                resolve_feature_indices(model_features, self.feature_names)
                if self.required_features
                else None
            )
            return

        self.feature_names = list(datamodule.feature_names)
        self.model_feature_indices = resolve_feature_indices(model_features, self.feature_names)
        _reject_undetectable_reorder(self.model_feature_indices, len(self.feature_names))
        if self.required_features:
            _reject_mismatched_train_columns(model_features, list(datamodule.features))

    def configure_optimizers(self) -> dict:
        """Build the optimiser, and the scheduler dict Lightning expects beside it."""
        optimizer = self._optimizer(params=self.parameters())
        if not self._scheduler:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": self._set_up_scheduler(optimizer)}

    def _set_up_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """Instantiate the scheduler partial and return it inside its Lightning dict.

        Schedulers that need the run length take it from the trainer, which knows it only inside
        ``configure_optimizers``.
        """
        scheduler_cfg = dict(self._scheduler)
        scheduler_fn = scheduler_cfg.pop("scheduler")
        params = inspect.signature(scheduler_fn).parameters
        kwargs = {}
        if "total_steps" in params:
            kwargs["total_steps"] = int(self.trainer.estimated_stepping_batches)
        elif "T_max" in params:
            kwargs["T_max"] = int(self.trainer.estimated_stepping_batches)
        scheduler_cfg["scheduler"] = scheduler_fn(optimizer=optimizer, **kwargs)
        return scheduler_cfg


def _reject_mismatched_train_columns(model_features: list[str], data_features: list[str]) -> None:
    """Reject a ``data.features`` list the full-width reduction indices do not describe.

    Those indices address the record's 16 columns, which is the width of the test and SCL batches.
    Training batches are ``data.features`` wide and reach a network with
    :attr:`JetTaggingModule.required_features` unreduced, so they have to be its own columns in its
    own order already.

    :raises ValueError: on any other feature list, naming the one the algorithm needs.
    """
    if resolve_feature_indices(model_features, data_features) != list(range(len(data_features))):
        raise ValueError(
            f"The algorithm reads {model_features}, but data.features is {data_features}, "
            "which reaches it unreduced during training. Set data.features to exactly "
            f"{model_features}."
        )


def _reject_undetectable_reorder(indices: list[int], n_columns: int) -> None:
    """Reject a layout whose reduction :meth:`JetTaggingModule.forward` cannot detect.

    forward() reduces a batch only when its width differs from ``nbr_feats``, so a network
    consuming every column in some other order would silently be fed the original order instead.
    """
    if len(indices) == n_columns and indices != list(range(n_columns)):
        raise ValueError(
            "The algorithm consumes every data column but in a different order, which "
            "forward() cannot detect: it reduces only on a width mismatch. Reorder "
            "data.features to match the algorithm's feature order."
        )
