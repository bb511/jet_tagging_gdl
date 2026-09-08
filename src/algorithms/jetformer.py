"""JetFormer jet tagger, a replica of the reference architecture (arXiv:2601.17215)."""

from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.jetformer import JetFormerNet


class JetFormer(JetTaggingModule):
    """Jet tagger reading the constituents as a token sequence with a CLS token."""

    def build_net(
        self,
        embed_dim: int,
        num_heads: int,
        num_transformers: int,
        normalization: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the network; the arguments are documented on :class:`JetFormerNet`."""
        return JetFormerNet(
            in_dim=self.hparams.nbr_feats,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformers=num_transformers,
            out_dim=self.hparams.output_dim,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
        )
