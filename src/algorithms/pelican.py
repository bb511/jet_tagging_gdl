"""PELICAN, wired into the shared training loop."""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.pelican import PELICANClassifier
from src.data.components.augmentation import get_norm_scale


class PELICAN(JetTaggingModule):
    """Jet tagger built on permutation-equivariant aggregators of the Minkowski products
    of the constituents (arXiv:2307.16506)."""

    # The network needs only the full four-momentum.
    required_features = ["px", "py", "pz", "e"]

    def build_net(
        self,
        n_hidden: int = 25,
        n_message: int = 15,
        n_blocks: int = 5,
        dropout: float = 0.025,
        nbar: float = 49.0,
        add_beams: bool = True,
    ) -> nn.Module:
        """Build the aggregator network; the arguments are in components/pelican.py."""
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))
        return PELICANClassifier(
            n_hidden=n_hidden,
            n_message=n_message,
            n_blocks=n_blocks,
            dropout=dropout,
            nbar=nbar,
            add_beams=add_beams,
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
