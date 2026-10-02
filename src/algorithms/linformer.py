"""Single-head Linformer jet tagger of arXiv:2510.24784, trained in float.

The network reads the constituent slots in the record's order, by p_T rank: its
projections over the slots have one weight per slot, so, unlike plain attention, it is
not invariant under a permutation of the constituents.
"""

from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.linformer import LinformerNet


class Linformer(JetTaggingModule):
    """Jet tagger reading the constituents as tokens with one Linformer block."""

    def build_net(
        self,
        embed_dim: int,
        kv_proj_dim: int,
        ff_dim: int,
        head_layers: list[int],
        aggregation: str = "sum_over_n",
        batchnorm: bool = True,
    ) -> nn.Module:
        """Build the network; the arguments are in components/linformer.py."""
        return LinformerNet(
            in_dim=self.hparams.nbr_feats,
            nconstituents=self.hparams.nconstituents,
            embed_dim=embed_dim,
            kv_proj_dim=kv_proj_dim,
            ff_dim=ff_dim,
            head_layers=head_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
            batchnorm=batchnorm,
        )
