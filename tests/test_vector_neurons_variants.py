import pytest
import torch

from src.algorithms.components.vector_neurons import VectorNeuronsClassifier

N_SLOTS = 16
# The state dict of the default network before the latency switches existed, which
# every float vn checkpoint of one block holds.
DEFAULT_KEYS = [
    "stream.norm.weight",
    "stream.norm.bias",
    "stream.norm.running_mean",
    "stream.norm.running_var",
    "stream.norm.num_batches_tracked",
    "stream.embed.weight",
    "stream.embed.bias",
    "stream.local.weight",
    "stream.local.bias",
    "stream.context.weight",
    "stream.context.bias",
    "tensors.weight",
    "tensors.left.weight",
    "tensors.right.weight",
    "blocks.0.linear.weight",
    "blocks.0.invariants.left.weight",
    "blocks.0.invariants.right.weight",
    "blocks.0.norm.weight",
    "blocks.0.norm.bias",
    "blocks.0.norm.running_mean",
    "blocks.0.norm.running_var",
    "blocks.0.norm.num_batches_tracked",
    "blocks.0.mlp.0.weight",
    "blocks.0.mlp.0.bias",
    "blocks.0.mlp.2.weight",
    "blocks.0.mlp.2.bias",
    "blocks.0.gate.linear.weight",
    "blocks.0.gate.linear.bias",
    "readout.left.weight",
    "readout.right.weight",
    "head_norm.weight",
    "head_norm.bias",
    "head_norm.running_mean",
    "head_norm.running_var",
    "head_norm.num_batches_tracked",
    "head.0.weight",
    "head.0.bias",
    "head.2.weight",
    "head.2.bias",
    "head.4.weight",
    "head.4.bias",
]
# (gate, mlp_layers, stream_channels, tensor_invariants) of the latency ablation's
# variants.
VARIANTS = [(False, 2, 16, 2), (False, 1, 16, 2), (False, 1, 0, 2), (False, 1, 0, 0)]


def _padded_jets(n_jets: int = 6, n_real: int = 11) -> torch.Tensor:
    # Rows [p_T GeV, eta, phi], the padded rows all zero and shuffled among the real.
    generator = torch.Generator().manual_seed(1)
    gauss = torch.randn(3, n_jets, n_real, generator=generator)
    pt, eta, phi = 40.0 * gauss[0].exp(), 0.3 * gauss[1], 0.3 * gauss[2]
    x = torch.zeros(n_jets, N_SLOTS, 3)
    x[:, :n_real] = torch.stack([pt, eta, phi], dim=-1)
    return x[:, torch.randperm(N_SLOTS, generator=generator)]


def test_the_defaults_build_the_network_of_every_vn_checkpoint() -> None:
    net = VectorNeuronsClassifier(N_SLOTS, [64, 32])

    assert list(net.state_dict()) == DEFAULT_KEYS


@pytest.mark.parametrize(
    "gate, mlp_layers, stream_channels, tensor_invariants", VARIANTS
)
def test_each_variant_builds_and_gives_finite_logits(
    gate: bool, mlp_layers: int, stream_channels: int, tensor_invariants: int
) -> None:
    torch.manual_seed(0)
    net = VectorNeuronsClassifier(
        N_SLOTS,
        [32],
        num_blocks=2,
        tensor_invariants=tensor_invariants,
        stream_channels=stream_channels,
        mlp_layers=mlp_layers,
        gate=gate,
    )
    keys = list(net.state_dict())

    with torch.no_grad():
        net.train()(_padded_jets())
        logits = net.eval()(_padded_jets())

    assert logits.shape == (6, 5) and torch.isfinite(logits).all()
    assert not any(".gate." in key for key in keys)
    assert ("stream.embed.weight" in keys) == (stream_channels > 0)
    tensors = [key for key in keys if key.startswith("tensors.")]
    assert bool(tensors) == (tensor_invariants > 0)
    for b in range(2):
        mlp = [key for key in keys if key.startswith(f"blocks.{b}.mlp.")]
        assert len(mlp) == 2 * mlp_layers
