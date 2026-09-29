"""L-GATr-slim, wired into the shared training loop."""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.lgatr_slim import LGATrSlimClassifier
from src.data.components.augmentation import get_norm_scale


class LGATrSlim(JetTaggingModule):
    """Jet tagger built on L-GATr-slim.

    This is the scalars-and-vectors Lorentz-equivariant transformer detailed in the
    following publication: arXiv:2512.17011.

    """
    # The network needs only the full four-momentum.
    required_features = ["px", "py", "pz", "e"]

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
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))
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

    def _on_features_resolved(self) -> None:
        """Cache the factors returning [px, py, pz, e] to GeV, in that column order."""
        scales = [
            get_norm_scale(self.norm_scales, name) for name in self.required_features
        ]
        self.denorm_scales.copy_(torch.tensor(scales))

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the normalisation of a batch and hand it over as four-momenta.

        :param x: (jets, constituents, 4), the columns [px, py, pz, e] as stored.
        :return: (jets, constituents, 4), the columns [E, px, py, pz] in GeV.
        """
        return (x * self.denorm_scales)[..., [3, 0, 1, 2]]
