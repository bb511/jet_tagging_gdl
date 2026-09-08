"""Flattened-input MLP jet tagger."""

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
        """Build the network; the arguments are documented in components/mlp.py."""
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
