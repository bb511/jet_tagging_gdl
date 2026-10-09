from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from lightning import Callback, LightningModule, Trainer
from torch import nn

from src.algorithms.bdt import BoostedTreesClassifier
from src.algorithms.components.bdt import BoostedTrees
from src.algorithms.components.deepsets import (
    AGGREGATIONS,
    PT_UNIT_GEV,
    RANK_CUTS,
    DeepSetsNet,
    JEDILinearNet,
    JetMoments,
    _masked_pool,
)
from src.algorithms.components.mlp import _dense
from src.algorithms.components.pelican import LinEq2to2
from src.algorithms.components.vector_neurons import (
    TensorInvariants,
    VectorNeuronsClassifier,
    _form,
    _form_matrix,
    _unit_lift,
)
from src.algorithms.utils.bitops import BitOps, count_ebops
from src.algorithms.utils.flops import count_flops
from src.data.components.augmentation import make_transform
from src.data.hls4ml_jet_level_datamodule import JetLevelDataModule
from src.symmetry import CLASS_NAMES_OF, GROUPS
from tests.conftest import (
    FOUR_MOMENTUM,
    JET_LEVEL_SIZES,
    MASS,
    MODELS,
    PT_ETA_PHI,
    PT_ETAREL_PHIREL,
    READS,
    RECORD_COLUMNS,
    StubDataModule,
    algorithm,
    attach,
    cylindrical_jets,
    exact_network,
    jets,
    labels,
    record_jets,
    with_random_gates,
)

# The exact probe groups of each embedded and moment row, the table of CLAUDE.md.
EXACT_GROUPS = {
    **dict.fromkeys(
        ("lorentznet", "pelican_lorentz", "lgatr_slim_lorentz", "lloca_lorentz"), GROUPS
    ),
    "lloca_so3": ("so2", "so3"),
    **dict.fromkeys(("lloca_z", "vn", "moments_only"), ("so2", "lorentz")),
    **dict.fromkeys(
        (
            "pelican",
            "lgatr_slim",
            "lloca_so2",
            "deepsets_moments",
            "jedi_linear_moments",
        ),
        ("so2",),
    ),
}
# The record's factors of [pt, etarel, phirel] at N = 16, so that the moments are
# de-normalised as a run does.
MOMENT_SCALES = torch.tensor([1559.19, 1.24234, 0.463250], dtype=torch.float64)
TREES = 10


class _Lambda(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


@pytest.mark.parametrize(
    "row, features",
    [(row, READS.get(row, FOUR_MOMENTUM)) for row in MODELS]
    + [("deepsets", PT_ETA_PHI), ("mlp", PT_ETA_PHI)],
)
def test_every_row_reads_its_own_columns_at_either_width(
    row: str, features: list[str]
) -> None:
    module = attach(algorithm(row, features=features), StubDataModule(features))
    # The momenta stand in for whatever the row reads: the network sees a width, not a
    # meaning. The test batches, every SCL batch and the probe's arrive at full width.
    narrow = jets(8, 8, 6)[..., : len(features)].contiguous()
    wide = jets(8, 8, 6, len(RECORD_COLUMNS), seed=2)
    wide[..., module.model_feature_indices] = narrow

    out = module.model_step((narrow, labels(8)))

    assert torch.isfinite(out["loss"]) and out["logits"].shape == (8, 5)
    assert count_ebops(module, narrow).unclassified == frozenset()
    with torch.no_grad():
        assert torch.allclose(module.eval()(wide), module(narrow), atol=1e-6)


@pytest.mark.parametrize("row", MODELS)
def test_every_row_tags_two_classes_and_refuses_a_class_count_mismatch(
    row: str,
) -> None:
    features = READS.get(row, FOUR_MOMENTUM)
    module = algorithm(row, features=features, output_dim=2)
    two_classes = StubDataModule(features, class_names=CLASS_NAMES_OF["top_tagging"])
    with pytest.raises(ValueError, match="output_dim"):
        attach(module, StubDataModule(features))
    attach(module, two_classes)
    batch = jets(8, 8, 6)[..., : len(features)].contiguous()

    out = module.model_step((batch, labels(8, n_classes=2)))

    assert torch.isfinite(out["loss"]) and out["logits"].shape == (8, 2)


@pytest.mark.parametrize(
    "row, features, match",
    [
        # forward() reduces on a width mismatch alone, blind to a full-width reorder.
        ("deepsets", list(reversed(RECORD_COLUMNS)), "different order"),
        ("lorentznet", ["px", "py", "pz"], "'e'"),
        ("vn", ["pt", "phi", "eta"], "exactly"),
        ("deepsets_moments", [*PT_ETAREL_PHIREL, "eta"], "exactly"),
        ("jedi_linear_moments", ["pt", "phirel", "etarel"], "exactly"),
    ],
)
def test_setup_refuses_a_feature_list_forward_would_misread(
    row: str, features: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        attach(algorithm(row), StubDataModule(features))


def _widened(
    narrow: dict[str, torch.Tensor], wide: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    # Each tensor of the narrower network fills the leading slice of its wider twin, so
    # a weight of the extra slot (the Linformer's projection rows) keeps its own draw.
    for name, value in narrow.items():
        wide[name][tuple(slice(0, n) for n in value.shape)] = value
    return wide


def _widened_vn(
    narrow: dict[str, torch.Tensor], wide: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    # From eight ranks to nine. The ranks are the input channels of the first block and
    # of the tensor invariants, and their scalars, two per rank, lead the inputs of the
    # first scalar MLP; an all-zero row leaves all three at zero, the batch norm at
    # initialisation mapping zero to zero. The tail moves up by one rank. The context
    # n / N falls by 8/9, so the two normalisation weights that read it, against running
    # means of zero, rise by 9/8.
    for name, value in narrow.items():
        if value.ndim == 0:
            continue
        if name.startswith(("net.blocks.0.norm.", "net.blocks.0.mlp.0.weight")):
            wide[name][..., :16] = value[..., :16]
            wide[name][..., 18:] = value[..., 16:]
        else:
            wide[name][..., : value.shape[-1]] = value
    for name in ("net.blocks.0.norm.weight", "net.head_norm.weight"):
        wide[name][-1] *= 9 / 8
    return wide


@pytest.mark.parametrize(
    "row",
    [
        "deepsets",
        "jedi_linear",
        "linformer",
        "lorentznet",
        "pelican",
        # The dense key-padding mask.
        "lgatr_slim",
        # The packed frames-net call and the key-padding mask: the frames-net's edge
        # convolution has no mask of its own, so a padded slot reaching it would move
        # the frame of every real constituent.
        "lloca_lorentz",
        "vn",
    ],
)
def test_a_padded_slot_changes_no_logit(row: str) -> None:
    # A padded slot embeds to a function of the first bias, not to zero, so only the
    # mask keeps it out: the wider network sees one more of them. The Linformer's
    # default sum_over_n divides by N, which differs by design.
    overrides = {"aggregation": "sum"} if row == "linformer" else {}
    narrow, wide = (algorithm(row, n, **overrides).eval() for n in (8, 9))
    widen = _widened_vn if row == "vn" else _widened
    if row == "vn":
        with_random_gates(narrow)
    wide.load_state_dict(widen(narrow.state_dict(), wide.state_dict()), strict=True)
    batch = (cylindrical_jets(4, 8, 6) if row == "vn" else jets(4, 8, 6)).float()
    padded = torch.cat([batch, torch.zeros(4, 1, batch.shape[-1])], dim=1)

    with torch.no_grad():
        assert torch.allclose(narrow(batch), wide(padded), atol=1e-5)


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_the_masked_pool_reads_the_real_slots_alone(aggregation: str) -> None:
    # The padded slots flooded with 1e6 would win every channel of the maximum and swamp
    # either sum. Jet 0 has a hole among its real slots. sum_over_n divides by the slot
    # count N for every jet, a constant the next dense layer absorbs.
    generator = torch.Generator().manual_seed(4)
    h = torch.randn(4, 8, 16, generator=generator, dtype=torch.float64)
    real = torch.zeros(4, 8, 1, dtype=torch.bool)
    real[:, :6] = True
    real[0, 2] = False
    expected = {
        "sum": (h * real).sum(dim=1),
        "max": h.masked_fill(~real, -torch.inf).amax(dim=1),
        "sum_over_n": (h * real).sum(dim=1) / 8,
    }

    pooled = _masked_pool(torch.where(real, h, 1e6), real, aggregation)

    assert torch.allclose(pooled, expected[aggregation], rtol=1e-12)


@pytest.mark.parametrize(
    "row",
    ["lorentznet", "pelican", "lgatr_slim", "lloca_lorentz", "vn", "moments_only"],
)
def test_a_denormalising_row_reads_a_normalised_batch_as_the_physical_one(
    row: str,
) -> None:
    columns = READS.get(row, FOUR_MOMENTUM)
    batch = (cylindrical_jets(4, 8, 6) if row in READS else jets(4, 8, 6)).float()
    scales = {name: 2.0 + i for i, name in enumerate(columns)}
    module = algorithm(row).eval()

    with torch.no_grad():
        attach(module, StubDataModule(columns, keep_full_features_at_test=False))
        physical = module(batch)
        attach(
            module, StubDataModule(columns, scales, keep_full_features_at_test=False)
        )
        normalised = module(batch / torch.tensor(list(scales.values())))

    assert torch.allclose(physical, normalised, atol=1e-3)


def _shift(row: str, group: str) -> float:
    # Relative to the logit scale: an absolute bar on a near-zero network says nothing.
    # The rows of the record's cylindrical columns read massless jets, on which a beam
    # boost shifts every eta rigidly; the four-momentum rows read massive ones, far
    # above LLoCa's mass floor, whose lift does not commute with a boost.
    columns = RECORD_COLUMNS if row in READS else FOUR_MOMENTUM
    batch = record_jets() if row in READS else jets(8, 16, 11).double() * 10.0
    network = exact_network(row)
    torch.manual_seed(1)
    transformed = make_transform(group, columns, None, 2.0)(batch)
    with torch.no_grad():
        logits = network(batch)
        return ((logits - network(transformed)).abs().max() / logits.abs().max()).item()


@pytest.mark.parametrize(
    "row, group", [(row, group) for row in EXACT_GROUPS for group in GROUPS]
)
def test_each_embedded_row_holds_exactly_its_probe_groups(row: str, group: str) -> None:
    # The exact groups sit at the float32 rounding of the batch and the transform
    # matrices, 1e-8 to 1e-7. At initialisation the broken groups of PELICAN and
    # L-GATr-slim shift the logits by 1e-5 to 1e-3, those of LLoCa and the vector
    # neurons by 0.1 to 4 times the largest logit and those of the moment rows by 7e-3
    # to 1.6 times: read against the SO(2) shift of the same network, not a fixed bar.
    if group in EXACT_GROUPS[row]:
        assert _shift(row, group) < (1e-6 if row in READS else 1e-4)
    else:
        assert _shift(row, group) > 1000 * _shift(row, "so2")


def test_compiling_keeps_the_eager_keys() -> None:
    # A torch.compile wrapper would prefix every key with _orig_mod., and eval.py, which
    # never compiles, could not load what a compiled training saved. Compiling is lazy,
    # so no graph is built here.
    eager = algorithm("lgatr_slim", compile=False)
    compiled = attach(algorithm("lgatr_slim", compile=True), StubDataModule())
    compiled.on_train_start()

    assert compiled.net._compiled_call_impl is not None
    assert list(compiled.state_dict()) == list(eager.state_dict())


def test_lloca_standardises_its_frames_net_on_the_first_training_batch() -> None:
    # eval.py reads the statistics back from the checkpoint, so the eval-mode passes
    # before the first training batch (the model summary, the sanity check) must leave
    # them unset, or the evaluated network would read its edges on another scale.
    module = algorithm("lloca_lorentz")
    block = module.net.framesnet.equivectors.block
    with torch.no_grad():
        module.eval()(jets(4, 8, 6))
        assert not block.edge_inited
        module.train()(jets(4, 8, 6))

    assert block.edge_inited and block.edge_std.item() != 1.0
    state = module.state_dict()
    assert all(
        f"net.framesnet.equivectors.block.edge_{s}" in state
        for s in ("inited", "mean", "std")
    )


def _ragged(multiplicities: list[int], n_slots: int = 16) -> torch.Tensor:
    batch = jets(len(multiplicities), n_slots, n_slots)
    for row, multiplicity in enumerate(multiplicities):
        batch[row, multiplicity:] = 0.0
    # A hole inside the leading slots of jet 0: a chunk width read off the real count
    # rather than off the mask would cut that jet short of its last real constituent.
    batch[0, 3] = 0.0
    return batch.double()


def test_pelican_chunking_is_exact() -> None:
    # Dropout off: its masks are drawn at the chunk shapes. A _track called once per
    # chunk would decay the running estimates four times faster than the statistics the
    # training mode normalises by.
    dense = algorithm("pelican", 16, n_chunks=1, dropout=0.0).double()
    chunked = algorithm("pelican", 16, n_chunks=4, dropout=0.0).double()
    chunked.load_state_dict(dense.state_dict(), strict=True)

    with torch.no_grad():
        batch = _ragged([16, 1, 9, 3, 14, 5, 12, 2])
        assert torch.allclose(dense(batch), chunked(batch), atol=1e-8)
        # Fewer jets than chunks leave empty pieces, which the FLOP pass at B = 2 meets.
        assert (
            chunked.eval()(jets(2, 16, 5)).shape == (2, 5) and chunked.flops_total > 0
        )
    reference = dict(dense.named_buffers())
    for name, value in chunked.named_buffers():
        assert torch.allclose(reference[name], value, atol=1e-8), name


def _stacked_basis(x: torch.Tensor, nobj: torch.Tensor) -> torch.Tensor:
    # The reference's eops_2_to_2 (ops[1] to ops[15]) written out from its own formulas
    # as a (B, C, 15, N, N) stack: the compact layer is checked against something other
    # than its own algebra. (B, C, N) -> (B, C, N, N) constant down a column or along a
    # row.
    inputs = x.permute(0, 3, 1, 2)
    n = inputs.shape[-1]
    down = lambda v: v.unsqueeze(2).expand(-1, -1, n, -1)  # noqa: E731
    along = lambda v: v.unsqueeze(3).expand(-1, -1, -1, n)  # noqa: E731
    diag = torch.diagonal(inputs, dim1=-2, dim2=-1)
    count = nobj.view(-1, 1, 1)
    sum_diag = diag.sum(2, keepdim=True) / count
    sum_rows, sum_cols = inputs.sum(3) / count, inputs.sum(2) / count
    sum_all = inputs.sum((2, 3)) / nobj.view(-1, 1) ** 2
    ops = [
        inputs,
        inputs.transpose(2, 3),
        torch.diag_embed(diag),
        down(diag),
        along(diag),
        torch.diag_embed(sum_cols),
        down(sum_cols),
        along(sum_cols),
        torch.diag_embed(sum_rows),
        down(sum_rows),
        along(sum_rows),
        sum_diag.unsqueeze(3).expand(-1, -1, n, n),
        torch.diag_embed(sum_diag.expand(-1, -1, n)),
        sum_all[..., None, None].expand(-1, -1, n, n),
        torch.diag_embed(sum_all.unsqueeze(-1).expand(-1, -1, n)),
    ]
    return torch.stack(ops, dim=2)


def test_the_pelican_pair_layer_is_the_reference_basis() -> None:
    # A pair tensor as the network builds one, zero on every padded pair, with a
    # multiplicity counting the real slots alone.
    torch.manual_seed(0)
    layer = LinEq2to2(3, 4, nbar=6.0).double()
    real = torch.zeros(2, 6, dtype=torch.bool)
    real[:, :4] = True
    pair_mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(-1)
    x = torch.where(pair_mask, torch.randn(2, 6, 6, 3).double(), 0.0)
    nobj = real.sum(-1, True).double()
    eye = torch.eye(6).double().view(1, 6, 6, 1)

    with torch.no_grad():
        # The layer keeps the ten multiplicity-scaled averages in weight slots 0 to 9
        # and the array and its diagonal, which carry no exponent, in 10 to 14: the
        # reference's ops[6] to ops[15], then ops[1] to ops[5].
        alphas = torch.cat([layer.alphas, torch.zeros(3, 5).double()], dim=1)
        scale = (nobj.view(-1, 1, 1, 1, 1) / 6.0) ** alphas.view(1, 3, 15, 1, 1)
        ops = _stacked_basis(x, nobj).roll(-5, dims=2) * scale
        out = torch.einsum("cbs,ncbij->nijs", layer.weight, ops) + layer.bias
        # Both biases go in before the activation, the diagonal one on the diagonal.
        out = F.leaky_relu(out + eye * layer.diag_bias, 0.01)
        expected = torch.where(pair_mask, out, 0.0)

        assert torch.allclose(expected, layer(x, pair_mask, nobj), atol=1e-10)


@pytest.mark.parametrize(
    "row, features, invariant",
    [
        ("lorentznet", FOUR_MOMENTUM, True),
        ("vn", PT_ETA_PHI, True),
        ("deepsets", FOUR_MOMENTUM, False),
    ],
)
def test_the_scl_vanishes_on_an_invariant_network_alone(
    row: str, features: list[str], invariant: bool
) -> None:
    # Under the SCL the batch arrives at full width, the partner is drawn on the
    # record's columns, and forward() reduces both. Massive jets: a massless constituent
    # puts LorentzNet's Minkowski norms at the float32 rounding of E^2.
    torch.manual_seed(0)
    module = algorithm(row, method="symm_loss", group="so2", features=features)
    attach(module, StubDataModule(features))

    out = module.model_step((record_jets(8, 8, 6, mass=MASS).float(), labels(8)))

    cross_entropy = F.cross_entropy(out["logits"], out["targets"])
    assert torch.allclose(out["loss"], cross_entropy + 0.1 * out["symm_loss"])
    relative = out["symm_loss"].item() / out["logits"].abs().max().item()
    # Zero up to the float32 rounding of the batch and of LorentzNet's Minkowski
    # products, 6e-5 of the largest logit here; DeepSets moves by 0.17.
    assert (relative < 1e-3, relative > 1e-2) == (invariant, not invariant), relative


def test_the_counters_reproduce_their_hand_counts() -> None:
    # JEDI-linear at N = 16 on four features: 64 F + 19 397 parameters at the published
    # point. Per jet, the multiplications are 16 x 4 x 64 (embed) + 2 x 16 x 64 x 64
    # (local, shared) + 64 x 64 (context) + 64 x 64 + 64 x 32 + 32 x 16 + 16 x 5 (head),
    # 146 000 in matrix products, which the FLOP counter prices at 2, plus 2 x 1024 by
    # the 0/1 masks of the two pools: 148 048 at 8 x 8 bits. The adds are the biases and
    # ReLUs of embed, local and shared (6144), of context (128) and of the head (229),
    # the residual (1024), the two pools' sums (2 x 960) and the mask's abs, sum and
    # compare (128): 9573 at 8 bits.
    module = algorithm("jedi_linear", nconstituents=16)
    # The CPU attention kernel on (jets, heads, L = S, E) = (2, 2, 6, 4): 144 scores
    # of 4 products in Q.K^T and 4 in A.V, the scaling of 4 x 12 rows of 4, a mask add
    # per score, and the softmax unpriced.
    x = torch.rand(2, 2, 6, 4)
    attention = _Lambda(lambda q: F.scaled_dot_product_attention(q, q, q))
    mask = torch.zeros(6, 6)
    masked = _Lambda(lambda q: F.scaled_dot_product_attention(q, q, q, attn_mask=mask))
    plain = BitOps(64 * (144 * 8 + 4 * 12 * 4), 144, frozenset())

    assert sum(p.numel() for p in module.parameters()) == 19_653
    assert module.flops_total == 2 * 146_000
    assert module.ebops_total == 64 * 148_048 + 8 * 9_573
    assert module.ebops_unpriced == 0.0
    assert count_ebops(attention, x) == plain
    assert count_ebops(masked, x) == plain._replace(ebops=plain.ebops + 8 * 144)
    assert count_flops(attention, x) == 2 * 2 * (4 * 6 * 6 * 4)
    noisy = _Lambda(lambda q: q + torch.rand_like(q)).train()
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    count_ebops(noisy, x)
    assert torch.equal(torch.rand(3), expected) and noisy.training


def _assert_committed(
    net: nn.Module,
    stacks: dict[str, nn.Sequential],
    committed: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> None:
    state, reference = net.state_dict(), nn.ModuleDict(stacks).state_dict()
    assert list(state) == list(reference)
    assert all(torch.equal(value, reference[name]) for name, value in state.items())
    with torch.no_grad():
        assert torch.equal(net(x), committed(x))


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_without_moments_the_parents_keep_their_committed_draws(
    aggregation: str,
) -> None:
    # The networks the DeepSets and JEDI-linear finals hold, rebuilt from their dense
    # stacks in the committed order from the same seed, and their forward passes
    # written out: under moments="none" the moments are a plain attribute.
    x = cylindrical_jets()
    real = x.abs().sum(dim=-1, keepdim=True) > 0
    kwargs = dict(in_dim=3, out_dim=5, aggregation=aggregation)

    def pool(h: torch.Tensor) -> torch.Tensor:
        return _masked_pool(h, real, aggregation)

    torch.manual_seed(0)
    deepsets = DeepSetsNet(**kwargs, phi_layers=[64, 64, 64], rho_layers=[64, 32])
    torch.manual_seed(0)
    phi, rho = _dense([3, 64, 64, 64], True), _dense([64, 64, 32, 5], False)
    stacks = {"phi": phi.double(), "rho": rho.double()}
    _assert_committed(deepsets.double(), stacks, lambda x: rho(pool(phi(x))), x)

    torch.manual_seed(0)
    jedi = JEDILinearNet(**kwargs, hidden=128, head_layers=[32, 32, 16]).double()
    torch.manual_seed(0)
    widths = {"embed": 3, "local": 128, "context": 128, "shared": 128}
    stacks = {name: _dense([n, 128], True).double() for name, n in widths.items()}
    stacks["head"] = _dense([128, 32, 32, 16, 5], False).double()

    def committed(x: torch.Tensor) -> torch.Tensor:
        h = stacks["embed"](x)
        h = stacks["local"](h) + stacks["context"](pool(h)).unsqueeze(1)
        return stacks["head"](pool(stacks["shared"](h)))

    _assert_committed(jedi, stacks, committed, x)


def _moment_jets(pt_max: float = 300.0) -> torch.Tensor:
    # Rows [p_T GeV, eta, phi] in float64, padded rows at random slots among the real
    # ones. Jets 0, 1 and 2 hold one, two and every slot.
    g = torch.Generator().manual_seed(0)
    pt = 1.0 + torch.rand(9, 12, generator=g, dtype=torch.float64) * pt_max
    angles = 0.3 * torch.randn(9, 12, 2, generator=g, dtype=torch.float64)
    n_real = torch.randint(1, 13, (9,), generator=g)
    n_real[:3] = torch.tensor([1, 2, 12])
    real = torch.rand(9, 12, generator=g).argsort(dim=1) < n_real[:, None]
    return torch.cat([pt.unsqueeze(-1), angles], dim=-1) * real.unsqueeze(-1)


def _raw_moments(x: torch.Tensor) -> torch.Tensor:
    module = JetMoments("rankcut").double()
    module.denorm_scales.copy_(MOMENT_SCALES)
    return module.raw(x / MOMENT_SCALES)


def _reference_moments(x: torch.Tensor, cut: int | None) -> torch.Tensor:
    rows = []
    for jet in x:
        ranked = sorted(
            (i for i in range(len(jet)) if jet[i, 0] > 0), key=lambda i: -jet[i, 0]
        )
        jet = jet[ranked[:cut]]
        a = jet[:, 0] / PT_UNIT_GEV
        d = (jet[:, None, 1:] - jet[None, :, 1:]).pow(2).sum(dim=-1)
        terms = [
            a.sum(),
            torch.einsum("i,j,ij->", a, a, d),
            torch.einsum("i,j,l,ij,jl->", a, a, a, d, d),
            torch.einsum("i,j,ij->", a, a, d**2),
            torch.einsum("i,j,l,ij,jl,li->", a, a, a, d, d, d),
        ]
        rows.append(torch.stack(terms))
    return torch.stack(rows)


def test_the_moments_match_their_quadratic_reference() -> None:
    x = _moment_jets()
    out = _raw_moments(x).unflatten(1, (len(RANK_CUTS["rankcut"]), 5))

    for k, cut in enumerate(RANK_CUTS["rankcut"]):
        expected = _reference_moments(x, cut)
        torch.testing.assert_close(out[:, k], expected, rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize(
    "device, dtype",
    [
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.float16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"),
        ),
    ],
)
def test_the_moments_stay_float32_under_autocast(
    device: str, dtype: torch.dtype
) -> None:
    # p reaches 50 here, so a cubic overflows fp16, and the 8-bit mantissa of bf16
    # cannot resolve the cancellations: under the experiments' 16-mixed the moments must
    # leave autocast. The CPU case catches a missing guard on any machine.
    x = _moment_jets(pt_max=1000.0)
    module = JetMoments("rankcut").to(device)
    module.denorm_scales.copy_(MOMENT_SCALES.float())
    with torch.autocast(device_type=device, dtype=dtype):
        out = module.raw((x / MOMENT_SCALES).float().to(device))
    expected = _raw_moments(x).to(device)
    # Relative to each column's largest value: the moments cancel between large terms.
    scale = expected.abs().amax(dim=0).clamp_min(1.0)

    assert out.dtype == torch.float32
    assert ((out.double() - expected).abs() / scale).max() < 1e-4


def test_the_vn_form_and_tensor_invariants_are_their_explicit_sums() -> None:
    # dR^2 = 0.3^2 + 0.4^2 = 0.25 and p_1 p_2 = 1.5, so B(u_1, u_2) = 0.375, the jet's
    # B(U, U) twice that, and each lift is null.
    p = torch.tensor([[3.0, 0.5]], dtype=torch.float64)
    u = p.unsqueeze(-1) * _unit_lift(
        *torch.tensor([[[0.2, -0.1]], [[1.0, 1.4]]]).double()
    )
    metric, jet = _form_matrix().double(), u.sum(dim=1)
    torch.manual_seed(0)
    module = TensorInvariants(5, 3).double()
    g = torch.Generator().manual_seed(2)
    q = torch.rand(5, generator=g, dtype=torch.float64) + 0.5
    q[-1] = 0.0
    eta, phi = 0.5 * torch.randn(2, 5, generator=g, dtype=torch.float64)
    unit = _unit_lift(eta, phi)
    dr2 = (eta[:, None] - eta) ** 2 + (phi[:, None] - phi) ** 2
    a, b = module.left.weight.detach(), module.right.weight.detach()
    c, d, e = module.weight.detach()
    triples = [
        torch.einsum("ki,kj,kl,i,j,l,ij,jl->k", a, c, b, q, q, q, dr2, dr2),
        torch.einsum("ki,kj,i,j,ij->k", c, d, q, q, dr2**2),
        torch.einsum("ki,kj,kl,i,j,l,ij,jl,li->k", c, d, e, q, q, q, dr2, dr2, dr2),
    ]

    with torch.no_grad():
        invariants = module((q[:, None] * unit)[None], unit[None])[0]
    assert torch.allclose(
        _form(u[:, 0], u[:, 1], metric), torch.tensor([0.375]).double()
    )
    assert torch.allclose(_form(jet, jet, metric), torch.tensor([0.75]).double())
    assert torch.allclose(_form(u, u, metric), torch.zeros_like(p), atol=1e-12)
    assert torch.allclose(invariants, torch.cat(triples), rtol=1e-12)


def test_the_vn_axis_offset_reads_the_leader_and_moves_with_the_jet() -> None:
    # Rows [p_T GeV, eta, phi, etarel, phirel] of six slots, the last two padded.
    torch.manual_seed(0)
    net = VectorNeuronsClassifier(
        nconstituents=6,
        head_layers=[8],
        vector_channels=4,
        scalar_channels=8,
        invariants=4,
        tensor_invariants=1,
        stream_channels=0,
        gate=False,
        axis_offset=True,
    )
    net = net.double().eval()
    g = torch.Generator().manual_seed(1)
    x = 0.3 * torch.randn(10, 6, 5, generator=g, dtype=torch.float64)
    x[..., 0] = 1.0 + 100.0 * torch.rand(10, 6, generator=g, dtype=torch.float64)
    x[:, 4:] = 0.0
    real = (x[..., :1] > 0).double()
    leader = x[torch.arange(10), x[..., 0].argmax(dim=1)]
    # The jet and its axis shifted together leave etarel and phirel as they are.
    shifted = x + real * torch.tensor([0.0, 0.7, -0.4, 0.0, 0.0]).double()
    moved = x + real * torch.tensor([0.0, 0.0, 0.0, 0.2, 0.0]).double()
    seen = []
    net.blocks[0].norm.register_forward_pre_hook(lambda _, args: seen.append(args[0]))

    with torch.no_grad():
        logits, logits_shifted, logits_moved = net(x), net(shifted), net(moved)
    s_in = seen[0].shape[1] - net.blocks[0].invariants.width - 1

    assert torch.equal(seen[0][:, s_in - 2 : s_in], leader[:, 3:5])
    assert torch.allclose(logits_shifted, logits, rtol=0.0, atol=1e-10)
    assert not torch.allclose(logits_moved, logits, rtol=0.0, atol=1e-6)


def _tree_jets() -> tuple[torch.Tensor, torch.Tensor]:
    # One-slot jets whose class shifts the first observable, which the trees separate.
    generator = torch.Generator().manual_seed(0)
    classes = torch.arange(200) % 5
    x = torch.randn(200, 1, 6, generator=generator)
    x[:, 0, 0] += 2.0 * classes
    return x, classes


def test_the_bdt_estimator_rides_the_state_dict() -> None:
    x, classes = _tree_jets()
    trees, restored = (
        BoostedTrees(n_class=5, trees_per_epoch=TREES, random_state=0) for _ in range(2)
    )
    trees.fit_more(x, classes)

    restored.load_state_dict(trees.state_dict())

    assert set(trees.state_dict()) == {"_extra_state"}
    assert restored.estimator.n_iter_ == TREES and torch.equal(restored(x), trees(x))
    # The search spells no limit 0: the sweeper writes a None choice as 'None'.
    no_limit = BoostedTrees(
        n_class=5, trees_per_epoch=TREES, random_state=0, max_depth=0
    )
    assert no_limit.estimator.max_depth is None


class _IterationsAtValidation(Callback):
    def __init__(self) -> None:
        self.seen: list[int] = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not trainer.sanity_checking:
            self.seen.append(pl_module.net.estimator.n_iter_)


def test_each_bdt_epoch_is_validated_on_its_own_trees(
    jet_level_shards: Path, tmp_path: Path
) -> None:
    # Lightning validates an epoch before on_train_epoch_end, so the trees are added on
    # its last batch for epoch k to be scored on (k + 1) trees_per_epoch iterations.
    module = BoostedTreesClassifier(name="bdt", nconstituents=1, nbr_feats=53)
    datamodule = JetLevelDataModule(shards_dir=str(jet_level_shards), batch_size=16)
    recorder = _IterationsAtValidation()
    trainer = Trainer(
        max_epochs=3,
        accelerator="cpu",
        default_root_dir=tmp_path,
        logger=False,
        callbacks=[recorder],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=datamodule)

    assert recorder.seen == [TREES, 2 * TREES, 3 * TREES]
    # One mock step per batch, without which ModelCheckpoint never saves.
    assert trainer.global_step == 3 * (JET_LEVEL_SIZES["train"] // 16)
