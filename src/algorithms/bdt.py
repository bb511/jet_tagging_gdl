"""Gradient-boosted trees on the jet-level observables, trained by the Lightning loop.

Each epoch adds ``trees_per_epoch`` boosting iterations to one warm-started estimator,
so that the networks' callbacks apply unchanged: the checkpoint keeps the best epoch by
``val/acc``, early stopping counts epochs, and Optuna reads ``val/acc_best``.
"""

import math

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.bdt import BoostedTrees


class BoostedTreesClassifier(JetTaggingModule):
    """A ``HistGradientBoostingClassifier`` in the place of a network.

    Training is manual and has no torch optimiser: for the None of
    ``configure_optimizers`` Lightning puts in its mock optimiser, whose step each batch
    takes, since ``ModelCheckpoint`` saves nothing while ``global_step`` stays at zero.
    The trees are fitted on the last batch of an epoch, because Lightning validates an
    epoch before ``on_train_epoch_end``. Validation and test run the base class's steps
    on the trees' log-probabilities. The trees run in scikit-learn, outside the torch
    operations that the FLOP and eBOP counters trace, so the costs are NaN.
    """

    def build_net(self, trees_per_epoch: int = 10, **tree_kwargs) -> nn.Module:
        """Build the estimator; *tree_kwargs* are arguments of scikit-learn's
        ``HistGradientBoostingClassifier``, at its defaults unless the config sets them,
        and ``max_depth`` 0 sets no limit (components/bdt.py)."""
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
        # The whole training split rather than the epoch's batches, a fresh permutation
        # less a random partial batch: the fit reads the earlier trees through bin
        # edges refitted on the jets it is given, which only identical jets reproduce.
        jets = self.trainer.datamodule.data_train
        self.net.fit_more(jets.X, jets.y.argmax(dim=1))
        # One batch: every tree read on every training jet would add a pass whose cost
        # grows with the trees, for figures that select nothing.
        super().training_step(batch, batch_idx)
