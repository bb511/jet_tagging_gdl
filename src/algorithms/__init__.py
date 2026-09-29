# Shared training logic for the jet-tagging architectures.

import inspect
from collections.abc import Callable

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from src.algorithms.losses.symmetry import SymmetryConsistencyLoss
from src.algorithms.utils.flops import record_flops
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

SUPPORTED_METHODS = ("augmentation", "symm_loss")


def resolve_feature_indices(
    model_features: list[str], feature_names: list[str]
) -> list[int]:
    """Feature indices of *model_features* inside *feature_names*.

    Names match bare or with the record's ``j1_`` prefix, so ``px`` finds ``j1_px``.
    """
    index_of = {name: idx for idx, name in enumerate(feature_names)}
    missing = [
        f for f in model_features if f not in index_of and f"j1_{f}" not in index_of
    ]
    if missing:
        raise ValueError(
            f"Feature {missing[0]!r} is absent from the data columns {feature_names}. "
            f"data.features must contain {', '.join(model_features)}."
        )
    return [index_of.get(f, index_of.get(f"j1_{f}")) for f in model_features]


class JetTaggingModule(LightningModule):
    """Classify jets into five originating particles, under either training method.

    Subclasses set `required_features` when the network needs specific features. These
    need to be present in the data, otherwise an error is thrown.
    """

    # Features the network needs, in its own order; None means "whatever the data give".
    required_features: list[str] | None = None

    # Filled by setup() from the datamodule.
    feature_names: list[str] | None = None
    model_feature_indices: list[int] | None = None
    norm_scales: dict[str, float] | None = None

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
        transform_kwargs: dict[str, float | bool] | None = None,
        optimizer: Callable | None = None,
        scheduler: dict | None = None,
        compile: bool = False,
        **net_kwargs,
    ) -> None:
        """
        :param nconstituents: constituents per jet.
        :param group: symmetry group of the SCL, read only under ``method="symm_loss"``.
        :param symm_loss_metric: metric that the symmetry loss component uses;
            the default is the L2 distance between the basic and transformed logits.
        :param transform_kwargs: additional arguments for the symmetry transformation
            that is applied to the data.
        :param optimizer: optimiser used to train the network, Hydra object.
        :param scheduler: scheduler for the learning rate, Lightning object.
        :param net_kwargs: additional arguments for the specific network that is being
            used, forwarded to the 'build_net' method of each network class.
        """
        super().__init__()
        # Optimiser and scheduler partials out of hparams.
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler"])
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {list(SUPPORTED_METHODS)}, got {method!r}."
            )
        self.transform_kwargs = dict(transform_kwargs or {})

        self._optimizer = optimizer
        self._scheduler = scheduler
        self.net = self.build_net(**net_kwargs)
        # The SCL term, which checks its group at construction; nothing penalises augmentation.
        self.symm_loss = None
        if method == "symm_loss":
            self.symm_loss = SymmetryConsistencyLoss(
                group, symm_loss_weight, symm_loss_metric
            )
        self._build_metrics(output_dim, method)

        self.example_input_array = torch.randn(2, nconstituents, nbr_feats)
        # Count the FLOPs of the model.
        record_flops(self, nconstituents, nbr_feats)

    def _build_metrics(self, output_dim: int, method: str) -> None:
        """Register the metric objects; the SCL trackers exist only under
        ``symm_loss``."""
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
        """Return the network this algorithm trains; every subclass implements it."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_net().")

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Last step before the network sees a batch; subclasses de-normalise or reorder
        here."""
        return x

    def _on_features_resolved(self) -> None:
        """Optional hook run at the end of `setup`, once the feature layout is known."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, const, features)."""
        x = x.to(torch.float32)
        if (
            self.model_feature_indices is not None
            and x.shape[-1] != self.hparams.nbr_feats
        ):
            x = x[:, :, self.model_feature_indices]
        return self.net(self._prepare_inputs(x))

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run one forward pass and compute the loss.

        :param batch: the pair ``(x, y)``, of shapes (jets, constituents, features) and
            one-hot (jets, classes).
        :return: dict of ``loss``, ``cls_loss``, ``logits``, ``preds``, ``targets``. If
            the method uses symmetry loss, you also include the SCL term added to the
            cross entropy term.
        """
        x, y = batch
        if y.ndim > 1 and y.shape[1] > 1:
            y = y.argmax(dim=1)

        logits = self(x)
        cls_loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        out = {
            "loss": cls_loss,
            "cls_loss": cls_loss,
            "logits": logits,
            "preds": preds,
            "targets": y,
        }

        if self.symm_loss is not None:
            partner_logits = self(self._partner_batch(x))
            out["loss"], out["symm_loss"] = self.symm_loss(
                cls_loss, logits, partner_logits
            )

        return out

    def _partner_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a transformed jet and return it, given an original jet.

        :raises RuntimeError: when `setup` has not set the columns of *x*.
        """
        if self.feature_names is None:
            raise RuntimeError(
                "feature_names is unset, so the columns of x cannot be named. Run under "
                "a trainer whose datamodule setup() has resolved the feature layout."
            )
        return self.symm_loss.transform(
            x, self.feature_names, self.norm_scales, **self.transform_kwargs
        )

    def on_train_start(self) -> None:
        """Clean up the sanity check values."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One optimisation step, logging the epoch means of the loss and the
        accuracy."""
        out = self.model_step(batch)
        self.train_loss(out["loss"])
        self.train_acc(out["preds"], out["targets"])
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        if self.train_symm_loss is not None:
            self.train_symm_loss(out["symm_loss"])
            self.log(
                "train/symm_loss", self.train_symm_loss, on_step=False, on_epoch=True
            )
        return out["loss"]

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Pass through a validation batch."""
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
        """Pass through a test batch."""
        out = self.model_step(batch)
        self.test_loss(out["loss"])
        self.test_acc(out["preds"], out["targets"])
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return out

    def on_train_epoch_end(self) -> None:
        """Log the epoch's training means."""
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
        """Resolve the feature layout from the datamodule, and compile the net."""
        if self.hparams.compile and stage == "fit":
            self._compile_net()
        # _trainer rather than trainer: the public property raises when the module stands alone.
        datamodule = getattr(self._trainer, "datamodule", None)
        if datamodule is None:
            return
        self._resolve_features(datamodule)
        self.norm_scales = datamodule.norm_scales
        self._on_features_resolved()

    def _compile_net(self) -> None:
        """Compile the network in place, so that a checkpoint keeps the eager key names that
        ``eval.py``, which never compiles, loads."""
        # A search driver fits trial after trial in one process, and every new architecture adds
        # graphs to the compiler's cache; past its recompile limit (8 entries, about four trials)
        # every later trial runs eager. A reset before each fit keeps every trial compiled.
        torch.compiler.reset()
        self.net.compile()

    def _resolve_features(self, datamodule) -> None:
        """Record which columns of the incoming batches the network actually uses.

        Under the SCL, and whenever the datamodule keeps all 16 record columns at test
        time, batches are wider than the network's input and the reduction indices are
        needed; otherwise only a network with `required_features` reorders anything.
        """
        model_features = self.required_features or list(datamodule.features)
        if self.hparams.method == "symm_loss" or datamodule.keep_full_features_at_test:
            self._resolve_full_width_features(model_features, datamodule)
            return

        self.feature_names = list(datamodule.features)
        self.model_feature_indices = (
            resolve_feature_indices(model_features, self.feature_names)
            if self.required_features
            else None
        )

    def _resolve_full_width_features(
        self, model_features: list[str], datamodule
    ) -> None:
        """Check that the features of the data exist and are in the expected order."""
        self.feature_names = list(datamodule.feature_names)
        indices = resolve_feature_indices(model_features, self.feature_names)
        if len(indices) == len(self.feature_names) and indices != list(
            range(len(indices))
        ):
            raise ValueError(
                "The algorithm consumes every data column but in a different order, which "
                "forward() cannot detect: it reduces only on a width mismatch. Reorder "
                "data.features to match the algorithm's feature order."
            )
        data_features = list(datamodule.features)
        train_indices = resolve_feature_indices(model_features, data_features)
        if self.required_features and train_indices != list(range(len(data_features))):
            raise ValueError(
                f"The algorithm reads {model_features}, but data.features is {data_features}, "
                "which reaches it unreduced during training. Set data.features to exactly "
                f"{model_features}."
            )
        self.model_feature_indices = indices

    def configure_optimizers(self) -> dict:
        """Build the optimiser, and the scheduler dict Lightning expects beside it."""
        optimizer = self._optimizer(params=self.parameters())
        if not self._scheduler:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": self._set_up_scheduler(optimizer),
        }

    def _set_up_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """Instantiate the scheduler partial and return it inside its Lightning dict.

        If a scheduler needs to know the number of steps in the whole training, this is
        set at the level of the trainer, not here.
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
