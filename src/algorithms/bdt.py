"""Gradient-boosted trees on the jet-level observables, inside lightning loop.

Each epoch adds ``trees_per_epoch`` boosting iterations to one warm-started estimator.
The networks' callbacks thus apply unchanged.
"""

import math
import pickle

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

from src.algorithms import JetTaggingModule

# The class probabilities are floored before the logarithm.
PROBABILITY_FLOOR = 1e-7


def _rows(x: torch.Tensor) -> np.ndarray:
    """The jets (jets, 1, features) as the (jets, features) array scikit-learn reads."""
    return x.detach().reshape(len(x), -1).cpu().numpy()


def _classifier(
    random_state: int, max_depth: int | None, tree_kwargs: dict
) -> HistGradientBoostingClassifier:
    """An unfitted estimator that each fit extends by its new ``max_iter``.

    Early stopping is off. Every fit ends at exactly ``max_iter`` iterations.
    """
    return HistGradientBoostingClassifier(
        max_iter=0,
        max_depth=max_depth or None,
        warm_start=True,
        early_stopping=False,
        random_state=random_state,
        **tree_kwargs,
    )


class BoostedTrees(nn.Module):
    """One gradient-boosted classifier, grown ``trees_per_epoch`` iterations at a time.

    Before its first fit the module returns zeros, the log-probabilities of the uniform
    prediction up to a constant.

    :param n_class: classes, the width of the output.
    :param random_state: seeds the subsample the bin edges are fitted on and the feature
        subsampling.
    :param max_depth: the depth limit of a tree; 0 or None sets none. 0 is the spelling
        a search can carry, since the Optuna sweeper writes a None choice as the string
        'None'.
    :param tree_kwargs: further arguments of ``HistGradientBoostingClassifier``.
    """

    def __init__(
        self,
        n_class: int,
        trees_per_epoch: int,
        random_state: int,
        max_depth: int | None = None,
        **tree_kwargs,
    ) -> None:
        super().__init__()
        self.n_class = n_class
        self.trees_per_epoch = trees_per_epoch
        self.estimator = _classifier(random_state, max_depth, tree_kwargs)

    @property
    def fitted(self) -> bool:
        return hasattr(self.estimator, "n_iter_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class log-probabilities (jets, n_class) of jets (jets, 1,
        features), in float32 on the device of *x*."""
        if not self.fitted:
            return x.new_zeros(len(x), self.n_class)
        probabilities = self.estimator.predict_proba(_rows(x))
        log_p = np.log(np.clip(probabilities, PROBABILITY_FLOOR, 1.0))
        return torch.from_numpy(log_p).to(device=x.device, dtype=torch.float32)

    def fit_more(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        """Add ``trees_per_epoch`` iterations, fitted on jets (jets, 1, features) and
        their class indices (jets,).

        Every call must pass the same jets in the same order. Each fit refits the bin
        edges on the jets it is given and scores the earlier trees on the binned jets;
        only identical jets give identical edges, under which the earlier trees read as
        they were grown and the iterations equal those of one fit of them all.
        """
        self.estimator.max_iter += self.trees_per_epoch
        self.estimator.fit(_rows(x), labels.cpu().numpy())

    def get_extra_state(self) -> bytes:
        return pickle.dumps(self.estimator)

    def set_extra_state(self, state: bytes) -> None:
        self.estimator = pickle.loads(state)


class BoostedTreesClassifier(JetTaggingModule):
    """A ``HistGradientBoostingClassifier`` in the place of a network.

    Training is manual and has no torch optimiser. This is a hacky way to have BDTs with
    consistent design. The trees are fitted on the last batch of an epoch, because
    Lightning validates an epoch before ``on_train_epoch_end``. Validation and test run
    the base class's steps on the trees' log-probabilities. The costs reported by the
    trees are NaN.
    """

    def build_net(self, trees_per_epoch: int = 10, **tree_kwargs) -> nn.Module:
        """Build the estimator.

        *tree_kwargs* are arguments of scikit-learn's
        ``HistGradientBoostingClassifier``, at its defaults unless the config sets them,
        and ``max_depth`` 0 sets no limit (:class:`BoostedTrees`).
        """
        self.automatic_optimization = False
        # The run's torch seed, so that the folds share the estimator's draws as the
        # networks share their initialisation.
        seed = torch.initial_seed() % 2**32
        return BoostedTrees(
            self.hparams.output_dim, trees_per_epoch, seed, **tree_kwargs
        )

    def _record_costs(self, nconstituents: int, nbr_feats: int) -> None:
        self.flops_total = self.ebops_total = self.ebops_unpriced = math.nan

    def configure_optimizers(self) -> None:
        return None

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Advance ``global_step`` and, on the last batch of the epoch, add its trees
        and log the training loss and accuracy on that batch."""
        self.optimizers().step()
        if not self.trainer.is_last_batch:
            return
        jets = self.trainer.datamodule.data_train
        self.net.fit_more(jets.X, jets.y.argmax(dim=1))
        super().training_step(batch, batch_idx)
