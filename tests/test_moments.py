import math
from collections.abc import Callable

import pytest
import torch
from torch import nn

from src.algorithms.components.deepsets import (
    AGGREGATIONS,
    PT_UNIT_GEV,
    RANK_CUTS,
    DeepSetsNet,
    JEDILinearNet,
    JetMoments,
    _dense,
    _masked_pool,
)
from src.algorithms.deepsets import DeepSetsMoments, JEDILinearMoments

# The record's factors at N = 16, so that the tests de-normalise as a run does.
SCALES = torch.tensor([1559.19, 1.24234, 0.463250], dtype=torch.float64)
CUTS = RANK_CUTS["rankcut"]
N_JETS, N_SLOTS = 9, 12
NETS = {
    "DeepSetsNet": dict(phi_layers=[64, 64, 64], rho_layers=[64, 32], dropout=0.0),
    "JEDILinearNet": dict(hidden=128, head_layers=[32, 32, 16]),
}
# The state-dict keys of the two networks at the sizes of NETS before the moments existed:
# the checkpoints of the DeepSets and JEDI-linear finals hold these, and eval.py loads them.
COMMITTED_KEYS = {
    "DeepSetsNet": [
        "phi.0.weight",
        "phi.0.bias",
        "phi.2.weight",
        "phi.2.bias",
        "phi.4.weight",
        "phi.4.bias",
        "rho.0.weight",
        "rho.0.bias",
        "rho.2.weight",
        "rho.2.bias",
        "rho.4.weight",
        "rho.4.bias",
    ],
    "JEDILinearNet": [
        "embed.0.weight",
        "embed.0.bias",
        "local.0.weight",
        "local.0.bias",
        "context.0.weight",
        "context.0.bias",
        "shared.0.weight",
        "shared.0.bias",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
        "head.4.weight",
        "head.4.bias",
        "head.6.weight",
        "head.6.bias",
    ],
}


def _jets(seed: int = 0, pt_max: float = 300.0) -> torch.Tensor:
    # Rows [p_T GeV, eta, phi] in float64. Jets 0, 1 and 2 hold one, two and every slot.
    g = torch.Generator().manual_seed(seed)
    pt = 1.0 + torch.rand(N_JETS, N_SLOTS, generator=g, dtype=torch.float64) * pt_max
    angles = 0.3 * torch.randn(N_JETS, N_SLOTS, 2, generator=g, dtype=torch.float64)
    x = torch.cat([pt.unsqueeze(-1), angles], dim=-1)
    n_real = torch.randint(1, N_SLOTS + 1, (N_JETS,), generator=g)
    n_real[:3] = torch.tensor([1, 2, N_SLOTS])
    # Padded rows at random slots, not only at the end.
    real = torch.rand(N_JETS, N_SLOTS, generator=g).argsort(dim=1) < n_real[:, None]
    return x * real.unsqueeze(-1)


def _shuffle(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    order = torch.rand(x.shape[:2], generator=g).argsort(dim=1)
    return x.gather(1, order.unsqueeze(-1).expand_as(x))


def _moments(weighting: str) -> JetMoments:
    module = JetMoments(weighting).double()
    module.denorm_scales.copy_(SCALES)
    return module


def _raw(module: JetMoments, x: torch.Tensor) -> torch.Tensor:
    return module.raw(x / SCALES).unflatten(1, (len(module.cuts), 5))


def _leading(jet: torch.Tensor, cut: int | None) -> list[int]:
    ranked = sorted(
        (i for i in range(len(jet)) if jet[i, 0] > 0), key=lambda i: -jet[i, 0]
    )
    return ranked if cut is None else ranked[:cut]


def _reference(x: torch.Tensor, cut: int | None) -> torch.Tensor:
    rows = []
    for jet in x:
        jet = jet[_leading(jet, cut)]
        a = jet[:, 0] / PT_UNIT_GEV
        d = (jet[:, None, 1:] - jet[None, :, 1:]).pow(2).sum(dim=-1)
        rows.append(
            torch.stack(
                [
                    a.sum(),
                    torch.einsum("i,j,ij->", a, a, d),
                    torch.einsum("i,j,l,ij,jl->", a, a, a, d, d),
                    torch.einsum("i,j,ij->", a, a, d**2),
                    torch.einsum("i,j,l,ij,jl,li->", a, a, a, d, d, d),
                ]
            )
        )
    return torch.stack(rows)


def _move(x: torch.Tensor, motion) -> torch.Tensor:
    eta, phi = motion(x[..., 1], x[..., 2])
    moved = torch.stack([x[..., 0], eta, phi], dim=-1)
    return torch.where(x[..., :1] > 0, moved, x)


def _rotation(eta: torch.Tensor, phi: torch.Tensor, angle: float = 0.9):
    c, s = math.cos(angle), math.sin(angle)
    return c * eta - s * phi, s * eta + c * phi


def test_closed_forms_match_explicit_sums():
    x = _jets()
    out = _raw(_moments("rankcut"), x)
    for k, cut in enumerate(CUTS):
        expected = _reference(x, cut)
        torch.testing.assert_close(out[:, k], expected, rtol=1e-10, atol=1e-9)


def test_moments_of_too_few_constituents_are_exact_zeros():
    out = _raw(_moments("rankcut"), _jets())
    assert torch.equal(out[:, 1, 1:], torch.zeros_like(out[:, 1, 1:]))
    assert torch.equal(out[:, 2, 4], torch.zeros_like(out[:, 2, 4]))
    # Jet 2 fills every slot, so nothing else is masked.
    assert (out[2, 2, 1:4] != 0).all() and (out[2, 3:, 4] != 0).all()


@pytest.mark.parametrize(
    "motion",
    [lambda e, p: (e + 0.7, p), lambda e, p: (e, p - 1.3), _rotation],
    ids=["eta_shift", "phi_shift", "rotation"],
)
def test_moments_invariant_under_plane_motions(motion):
    module, x = _moments("rankcut"), _jets()
    moved = _raw(module, _move(x, motion))
    torch.testing.assert_close(moved, _raw(module, x), rtol=1e-9, atol=1e-9)


def test_moments_invariant_under_permutations():
    module, x = _moments("rankcut"), _jets()
    for seed in range(3):
        shuffled = _raw(module, _shuffle(x, seed))
        torch.testing.assert_close(shuffled, _raw(module, x), rtol=1e-12, atol=1e-12)


def test_rank_cuts_select_the_leading_real_constituents():
    rankcut, uniform, x = _moments("rankcut"), _moments("uniform"), _jets()
    for seed in range(3):
        shuffled = _shuffle(x, seed)
        out = _raw(rankcut, shuffled)
        for k, cut in enumerate(CUTS):
            kept = torch.zeros_like(shuffled)
            for j, jet in enumerate(shuffled):
                rows = _leading(jet, cut)
                kept[j, rows] = jet[rows]
            expected = _raw(uniform, _shuffle(kept, seed + 10))[:, 0]
            torch.testing.assert_close(out[:, k], expected, rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize(
    "device, dtype",
    [
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.float16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs a GPU"
            ),
        ),
    ],
)
def test_moments_stay_float32_under_autocast(device: str, dtype: torch.dtype):
    # p reaches 50 here, so a cubic overflows fp16, and the 8-bit mantissa of bf16 cannot
    # resolve the cancellations: under the 16-mixed of the experiments the moments must
    # leave autocast. The CPU case catches a missing guard on any machine.
    x = _jets(pt_max=1000.0)
    module = JetMoments("rankcut").to(device)
    module.denorm_scales.copy_(SCALES.float())
    with torch.autocast(device_type=device, dtype=dtype):
        out = module.raw((x / SCALES).float().to(device))
    assert out.dtype == torch.float32
    expected = _raw(_moments("rankcut"), x).flatten(1).to(device)
    # Relative to each column's largest value: the moments cancel between large terms.
    scale = expected.abs().amax(dim=0).clamp_min(1.0)
    assert ((out.double() - expected).abs() / scale).max() < 1e-4


@pytest.mark.parametrize(
    "moments,only",
    [("uniform", False), ("rankcut", False), ("uniform", True), ("rankcut", True)],
)
def test_deepsets_variants_widen_rho(moments, only):
    phi_layers = None if only else [16, 8]
    net = DeepSetsNet(3, phi_layers, [8], 5, moments=moments, moments_only=only)
    net = net.double()
    assert hasattr(net, "phi") != only
    width = net.moments.width + (1 if only else 8)
    assert net.rho[0].in_features == width
    assert net(_jets() / SCALES).shape == (N_JETS, 5)


@pytest.mark.parametrize("moments", ["uniform", "rankcut"])
def test_jedi_linear_variants_widen_head(moments):
    net = JEDILinearNet(3, 16, [8], 5, moments=moments).double()
    assert net.head[0].in_features == 16 + net.moments.width
    assert net(_jets() / SCALES).shape == (N_JETS, 5)


def _assert_committed(
    net: nn.Module,
    stacks: dict[str, nn.Sequential],
    x: torch.Tensor,
    committed: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    # The same keys, the same draws from the same seed, and the same logits.
    state = net.state_dict()
    reference = nn.ModuleDict(stacks).state_dict()
    assert list(state) == COMMITTED_KEYS[type(net).__name__] == list(reference)
    assert all(torch.equal(value, reference[key]) for key, value in state.items())
    with torch.no_grad():
        assert torch.equal(net(x), committed(x))


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_without_moments_the_networks_compute_what_they_did(aggregation):
    # The committed networks rebuilt from their stacks, drawn in the committed order from
    # the same seed, and their forward passes written out.
    x = _jets() / SCALES
    real = x.abs().sum(dim=-1, keepdim=True) > 0
    kwargs = dict(in_dim=3, out_dim=5, aggregation=aggregation)

    def pool(h: torch.Tensor) -> torch.Tensor:
        return _masked_pool(h, real, aggregation)

    torch.manual_seed(0)
    deepsets = DeepSetsNet(**kwargs, **NETS["DeepSetsNet"]).double()
    torch.manual_seed(0)
    phi = _dense([3, 64, 64, 64], True).double()
    rho = _dense([64, 64, 32, 5], False).double()
    stacks = {"phi": phi, "rho": rho}
    _assert_committed(deepsets, stacks, x, lambda x: rho(pool(phi(x))))

    torch.manual_seed(0)
    jedi = JEDILinearNet(**kwargs, **NETS["JEDILinearNet"]).double()
    torch.manual_seed(0)
    widths = {"embed": 3, "local": 128, "context": 128, "shared": 128}
    stacks = {name: _dense([n, 128], True).double() for name, n in widths.items()}
    stacks["head"] = _dense([128, 32, 32, 16, 5], False).double()

    def committed(x: torch.Tensor) -> torch.Tensor:
        h = stacks["embed"](x)
        context = stacks["context"](pool(h))
        h = stacks["local"](h) + context.unsqueeze(1)
        return stacks["head"](pool(stacks["shared"](h)))

    _assert_committed(jedi, stacks, x, committed)


def test_a_deepsets_without_phi_reads_the_moments_alone_and_refuses_otherwise():
    net = DeepSetsNet(3, None, [8], 5, moments="uniform", moments_only=True)
    net = net.double().eval()
    x = _jets() / SCALES
    multiplicity = (x.abs().sum(dim=-1) > 0).sum(dim=1, keepdim=True).double() / N_SLOTS

    assert [name for name, _ in net.named_children()] == ["moments", "rho"]
    with torch.no_grad():
        expected = net.rho(torch.cat([net.moments(x), multiplicity], dim=-1))
        torch.testing.assert_close(net(x), expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="phi_layers"):
        DeepSetsNet(3, None, [8], 5, moments="uniform")
    with pytest.raises(ValueError, match="phi_layers"):
        DeepSetsNet(3, [8], [8], 5, moments="uniform", moments_only=True)
    with pytest.raises(ValueError, match="moments_only needs a moment set"):
        DeepSetsNet(3, None, [8], 5, moments_only=True)


def test_the_moment_rows_need_moments_and_jedi_linear_has_no_control():
    # Under "none" a moment row would be its parent under another name, and the control
    # is a DeepSets without phi, so JEDI-linear takes no moments_only.
    common = dict(nconstituents=N_SLOTS, nbr_feats=3, moments="none")
    with pytest.raises(ValueError, match="moment set"):
        DeepSetsMoments(
            name="deepsets_moments", phi_layers=[8], rho_layers=[8], **common
        )
    with pytest.raises(TypeError, match="moments_only"):
        JEDILinearMoments(
            name="jedi_linear_moments",
            nconstituents=N_SLOTS,
            nbr_feats=3,
            hidden=8,
            head_layers=[8],
            moments="uniform",
            moments_only=True,
        )
