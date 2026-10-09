"""L-GATr-slim, wired into the shared training loop."""

from torch import nn

from src.algorithms import FourMomentumModule
from src.algorithms.components.lgatr_slim import LGATrSlimClassifier


class LGATrSlim(FourMomentumModule):
    """Jet tagger built on L-GATr-slim.

    This is the scalars-and-vectors Lorentz-equivariant transformer detailed in the
    following publication: arXiv:2512.17011.
    """

    def build_net(
        self,
        num_blocks: int = 12,
        hidden_v_channels: int = 32,
        hidden_s_channels: int = 96,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        units: float = 20.0,
        add_beams: bool = True,
        add_time: bool = True,
    ) -> nn.Module:
        """Build the transformer; the arguments are in components/lgatr_slim.py."""
        self._register_denorm_scales()
        return LGATrSlimClassifier(
            num_blocks=num_blocks,
            hidden_v_channels=hidden_v_channels,
            hidden_s_channels=hidden_s_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            units=units,
            add_beams=add_beams,
            add_time=add_time,
            n_class=self.hparams.output_dim,
        )
