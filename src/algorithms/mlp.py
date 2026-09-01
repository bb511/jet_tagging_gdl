"""Flattened-input MLP jet tagger.

The constituents are read as one long vector, so the model is not even invariant under their
permutation. It is the floor against which the set-based and attention-based architectures, and the
two ways of teaching them a symmetry, are read.
"""

from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.mlp import MLP


class MLPClassifier(JetTaggingModule):
    """Jet tagger that flattens the constituent grid and reads it with a plain MLP."""

    def build_net(
        self,
        hidden_layers: list[int],
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> nn.Module:
        """Build the network; the arguments are documented on :class:`MLP`.

        The input width is the flattened grid, ``nconstituents * nbr_feats``, so the first layer
        grows with the constituent count while the widths below do not.
        """
        return nn.Sequential(
            nn.Flatten(1),
            MLP(
                in_dim=self.hparams.nconstituents * self.hparams.nbr_feats,
                nodes=hidden_layers,
                out_dim=self.hparams.output_dim,
                batchnorm=batchnorm,
                activation=activation,
            ),
        )
