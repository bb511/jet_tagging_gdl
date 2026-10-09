"""PELICAN, wired into the shared training loop."""

from torch import nn

from src.algorithms import FourMomentumModule
from src.algorithms.components.pelican import PELICANClassifier


class PELICAN(FourMomentumModule):
    """Jet tagger built on permutation-equivariant aggregators of the Minkowski products
    of the constituents (arXiv:2307.16506)."""

    def build_net(
        self,
        n_hidden: int = 25,
        n_message: int = 15,
        n_blocks: int = 5,
        dropout: float = 0.025,
        nbar: float = 49.0,
        add_beams: bool = True,
        n_chunks: int = 1,
    ) -> nn.Module:
        """Build the aggregator network; the arguments are in components/pelican.py."""
        self._register_denorm_scales()
        return PELICANClassifier(
            n_hidden=n_hidden,
            n_message=n_message,
            n_blocks=n_blocks,
            dropout=dropout,
            nbar=nbar,
            add_beams=add_beams,
            n_chunks=n_chunks,
            n_class=self.hparams.output_dim,
        )
