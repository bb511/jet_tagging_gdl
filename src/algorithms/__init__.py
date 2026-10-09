# Shared training logic for the jet-tagging architectures.

import inspect
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from src.algorithms.losses.symmetry import SymmetryConsistencyLoss
from src.algorithms.utils.bitops import count_ebops_per_jet
from src.algorithms.utils.flops import count_flops_per_jet
from src.data.components.augmentation import (
    get_norm_scale,
    make_transform,
    resolve_feature_indices,
)
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class JetTaggingModule(LightningModule):
    """Classify jets into five originating particles, under either training method.

    Subclasses set `required_features` when the network needs specific features. These
    need to be present in the data, otherwise an error is thrown.
    """

    # Features the network needs, in its own order; None means "whatever the data give".
    required_features: list[str] | None = None

    # Filled by setup() from the datamodule; the transform under the SCL alone.
    feature_names: list[str] | None = None
    model_feature_indices: list[int] | None = None
    norm_scales: dict[str, float] | None = None
    _symm_transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    # Set by _record_costs at construction, per fully occupied jet.
    flops_total: float
    ebops_total: float
    ebops_unpriced: float

    def __init__(
        self,
        name: str,
        nconstituents: int,
        nbr_feats: int,
        output_dim: int = 5,
        method: str = "augmentation",
        group: str = "none",
        symm_loss_weight: float = 0.1,
        transform_kwargs: dict[str, float | bool] | None = None,
        optimizer: Callable | None = None,
        scheduler: dict | None = None,
        compile: bool = False,
        **net_kwargs,
    ) -> None:
        """
        :param nconstituents: constituents per jet.
        :param group: symmetry group of the SCL, read only under ``method="symm_loss"``.
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
        self.transform_kwargs = dict(transform_kwargs or {})

        self._optimizer = optimizer
        self._scheduler = scheduler
        self.net = self.build_net(**net_kwargs)
        # The SCL term, which checks its group at construction; nothing penalises augmentation.
        self.symm_loss = None
        if method == "symm_loss":
            self.symm_loss = SymmetryConsistencyLoss(group, symm_loss_weight)
        self._build_metrics(output_dim, method)

        self.example_input_array = torch.randn(2, nconstituents, nbr_feats)
        self._record_costs(nconstituents, nbr_feats)

    def _record_costs(self, nconstituents: int, nbr_feats: int) -> None:
        """Set ``flops_total``, ``ebops_total`` and ``ebops_unpriced``.

        The FLOP pass draws its example and any dropout mask from the global stream, so
        it is part of every run's initialisation stream; the eBOP pass leaves the stream
        untouched.
        """
        self.flops_total = count_flops_per_jet(self, nconstituents, nbr_feats)
        bitops = count_ebops_per_jet(self, nconstituents, nbr_feats)
        self.ebops_total, self.ebops_unpriced = bitops.ebops, bitops.unpriced

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
        return self.net(self._network_inputs(x))

    def _network_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return what the network reads of a batch: in float32, reduced to the
        network's columns when it arrives at full width, and prepared."""
        x = x.to(torch.float32)
        if (
            self.model_feature_indices is not None
            and x.shape[-1] != self.hparams.nbr_feats
        ):
            x = x[:, :, self.model_feature_indices]
        return self._prepare_inputs(x)

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run one forward pass and compute the loss.

        :param batch: the pair ``(x, y)``, of shapes (jets, constituents, features) and
            one-hot (jets, classes).
        :return: dict of ``loss``, ``logits``, ``preds``, ``targets`` and, under the
            SCL, ``symm_loss``, in which case ``loss`` is the cross entropy plus the
            weighted SCL term.
        """
        x, y = batch
        y = y.argmax(dim=1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        out = dict(loss=loss, logits=logits, preds=logits.argmax(dim=1), targets=y)
        if self.symm_loss is not None:
            partner_logits = self(self._partner_batch(x))
            out["loss"], out["symm_loss"] = self.symm_loss(loss, logits, partner_logits)
        return out

    def _partner_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Return *x* with one transformation of the SCL's group drawn per jet.

        :raises RuntimeError: when `setup` has not set the columns of *x*.
        """
        if self._symm_transform is None:
            raise RuntimeError(
                "feature_names is unset, so the columns of x cannot be named. Run under "
                "a trainer whose datamodule setup() has resolved the feature layout."
            )
        return self._symm_transform(x)

    def on_train_start(self) -> None:
        """Clean up the sanity check values, and compile the net if asked."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        if self.hparams.compile:
            self._compile_net()

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
        """Resolve the feature layout from the datamodule and, under the SCL, build the
        transform of its partner batches, which draws only when called."""
        # _trainer rather than trainer: the public property raises when the module stands alone.
        datamodule = getattr(self._trainer, "datamodule", None)
        if datamodule is None:
            return
        self._resolve_features(datamodule)
        self.norm_scales = datamodule.norm_scales
        if self.symm_loss is not None:
            group = self.symm_loss.group
            self._symm_transform = make_transform(
                group, self.feature_names, self.norm_scales, **self.transform_kwargs
            )
        self._on_features_resolved()

    def _compile_net(self) -> None:
        """Compile the network in place, so that a checkpoint keeps the eager key names
        that ``eval.py``, which never compiles, loads."""
        # At train start, not in setup(): Lightning's model summary runs one forward under
        # FlopCounterMode at fit start, and a compiled call made under that mode leaves the module
        # call marked to run eager for the rest of the process. A search driver also fits trial
        # after trial in one process, and every new architecture adds graphs to the compiler's
        # cache; past its recompile limit (8 entries, about four trials) every later trial runs
        # eager. The reset clears both.
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
        scheduler_cfg["scheduler"] = scheduler_fn(optimizer=optimizer, **kwargs)
        return scheduler_cfg


def denorm_factors(
    norm_scales: dict[str, float] | None, names: Sequence[str]
) -> torch.Tensor:
    """Return the factors taking the stored columns *names* back to physical units."""
    return torch.tensor([get_norm_scale(norm_scales, name) for name in names])


class DenormalisedModule(JetTaggingModule):
    """A row whose network reads its `required_features` in physical units.

    Each subclass's ``build_net`` calls `_register_denorm_scales` before it builds the
    network; `setup` then fills the factors from the datamodule's ``norm_scales``.
    """

    def _register_denorm_scales(self) -> None:
        """Register ``denorm_scales``, ones until the features are resolved."""
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))

    def _on_features_resolved(self) -> None:
        """Cache the factors returning `required_features` to physical units."""
        self.denorm_scales.copy_(
            denorm_factors(self.norm_scales, self.required_features)
        )

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the normalisation of a batch, keeping its column order."""
        return x * self.denorm_scales


class FourMomentumModule(DenormalisedModule):
    """A row whose network reads the four-momentum alone."""

    required_features = ["px", "py", "pz", "e"]

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the normalisation of a batch and hand it over as four-momenta.

        :param x: (jets, constituents, 4), the columns [px, py, pz, e] as stored.
        :return: (jets, constituents, 4), the columns [E, px, py, pz] in GeV.
        """
        return super()._prepare_inputs(x)[..., [3, 0, 1, 2]]
