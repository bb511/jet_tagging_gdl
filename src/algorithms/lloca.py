"""LLoCa, wired into the shared training loop."""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.lloca import LLoCaClassifier
from src.data.components.augmentation import get_norm_scale


class LLoCa(JetTaggingModule):
    """Jet tagger built on the LLoCa-Transformer, a transformer made equivariant by
    learned local frames (arXiv:2505.20280, arXiv:2508.14898)."""

    # The network needs only the full four-momentum.
    required_features = ["px", "py", "pz", "e"]

    def build_net(
        self,
        frames: str = "lorentz",
        num_blocks: int = 8,
        num_heads: int = 8,
        attn_scalars: int = 8,
        attn_vectors: int = 2,
        mlp_factor: int = 2,
        dropout: float = 0.0,
        framesnet_hidden: int = 32,
        mass_reg: float = 2.5e-4,
        units: float = 20.0,
    ) -> nn.Module:
        """Build the network; the arguments are in components/lloca.py."""
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))
        return LLoCaClassifier(
            frames=frames,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn_scalars=attn_scalars,
            attn_vectors=attn_vectors,
            mlp_factor=mlp_factor,
            dropout=dropout,
            framesnet_hidden=framesnet_hidden,
            mass_reg=mass_reg,
            units=units,
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
