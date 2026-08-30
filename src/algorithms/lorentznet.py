# LorentzNet as a jet-tagging algorithm: the Lorentz-invariant baseline of this study.
"""LorentzNet, wired into the shared training loop.

The network reads four-momenta in GeV, not the normalised columns the datamodule stores, and in the
order [E, px, py, pz]. Both changes happen in the algorithm's ``_prepare_inputs``, so the rest of
the pipeline handles this algorithm like any other. Its logits are invariant under the Lorentz
group by construction, which makes it the reference against which the augmented and SCL-trained
networks are read.
"""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.lorentznet import LorentzNetGraph
from src.data.components.augmentation import get_norm_scale


class LorentzNet(JetTaggingModule):
    """Jet tagger built on Lorentz Group Equivariant Blocks (arXiv:2201.08187)."""

    # The network needs the full four-momentum, and reads no other column.
    required_features = ["px", "py", "pz", "e"]

    def build_net(
        self,
        n_hidden: int = 72,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the graph network, and the buffer that returns a batch to GeV.

        :param n_hidden: width of the node scalars inside the blocks.
        :param n_layers: number of blocks.
        :param c_weight: step size of the coordinate updates.
        :param dropout: dropout rate in the classification head.
        """
        # Registered here rather than in an __init__ the subclass does not have: the base
        # class collects these arguments by walking the frame of save_hyperparameters.
        # Ones until setup() reads the datamodule, and persistent, so that a checkpoint
        # carries the scales its weights were trained with.
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))
        return LorentzNetGraph(
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_class=self.hparams.output_dim,
            c_weight=c_weight,
            dropout=dropout,
            nconstituents=self.hparams.nconstituents,
        )

    def _on_features_resolved(self) -> None:
        """Cache the factors returning [px, py, pz, e] to GeV, in that column order."""
        scales = [get_norm_scale(self.norm_scales, name) for name in self.required_features]
        self.denorm_scales.copy_(torch.tensor(scales))

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the normalisation of a batch and hand it over as four-momenta.

        :param x: (jets, constituents, 4), the columns [px, py, pz, e] as stored.
        :return: (jets, constituents, 4), the columns [E, px, py, pz] in GeV. The Minkowski
            invariants the network builds are dimensionful, so a normalised batch would put them at
            the wrong scale rather than merely rescale them.
        """
        return (x * self.denorm_scales)[..., [3, 0, 1, 2]]
