"""Vector neurons, wired into the shared training loop."""

from torch import nn

from src.algorithms import DenormalisedModule
from src.algorithms.components.vector_neurons import VectorNeuronsClassifier


class VectorNeurons(DenormalisedModule):
    """Jet tagger built on vector neurons (arXiv:2104.12229).

    The inputs are lifted from (p_T, eta, phi) to own 4-vectors, exact under the
    Euclidean group of the (eta, phi) plane.
    """

    # Absolute angles: the network measures them from its leading constituent itself.
    required_features = ["pt", "eta", "phi"]

    def build_net(
        self,
        head_layers: list[int],
        num_blocks: int = 1,
        vector_channels: int = 16,
        scalar_channels: int = 64,
        invariants: int = 16,
        tensor_invariants: int = 2,
        stream_channels: int = 16,
        mlp_layers: int = 2,
        gate: bool = True,
        dropout: float = 0.0,
        units: float = 20.0,
    ) -> nn.Module:
        """Build the network; the arguments are in components/vector_neurons.py."""
        self._register_denorm_scales()
        return VectorNeuronsClassifier(
            nconstituents=self.hparams.nconstituents,
            head_layers=head_layers,
            num_blocks=num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            tensor_invariants=tensor_invariants,
            stream_channels=stream_channels,
            mlp_layers=mlp_layers,
            gate=gate,
            dropout=dropout,
            units=units,
            n_class=self.hparams.output_dim,
        )
