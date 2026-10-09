"""LLoCa, wired into the shared training loop."""

from torch import nn

from src.algorithms import FourMomentumModule
from src.algorithms.components.lloca import LLoCaClassifier


class LLoCa(FourMomentumModule):
    """Jet tagger built on the LLoCa-Transformer, a transformer made equivariant by
    learned local frames (arXiv:2505.20280, arXiv:2508.14898)."""

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
        framesnet_dropout: float = 0.0,
        mass_reg: float = 2.5e-4,
        units: float = 20.0,
    ) -> nn.Module:
        """Build the network; the arguments are in components/lloca.py."""
        self._register_denorm_scales()
        return LLoCaClassifier(
            frames=frames,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn_scalars=attn_scalars,
            attn_vectors=attn_vectors,
            mlp_factor=mlp_factor,
            dropout=dropout,
            framesnet_hidden=framesnet_hidden,
            framesnet_dropout=framesnet_dropout,
            mass_reg=mass_reg,
            units=units,
            n_class=self.hparams.output_dim,
        )
