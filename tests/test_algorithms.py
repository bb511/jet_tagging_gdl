import math
from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.algorithms import JetTaggingModule
from src.algorithms.components.deepsets import _masked_pool
from src.algorithms.components.linformer import AGGREGATIONS as LINFORMER_AGGREGATIONS
from src.algorithms.components.linformer import LinformerNet, _pool
from src.algorithms.components.pelican import LinEq2to2
from src.algorithms.components.vector_neurons import (
    SplitGate,
    TensorInvariants,
    VectorNeuronsClassifier,
    _form,
    _form_matrix,
    _unit_lift,
)
from src.data.components.augmentation import (
    compute_eta,
    compute_phi,
    compute_pt,
    make_transform,
)
from src.symmetry import GROUPS
from tests.conftest import (
    BASELINE_ONLY,
    EMBEDDED,
    FOUR_MOMENTUM,
    MODELS,
    RECORD_COLUMNS,
    ROOT,
    StubDataModule,
    jets,
    labels,
)

# save_hyperparameters collects these through Lightning's frame walk; without them a checkpoint
# cannot rebuild the network it holds.
ARCHITECTURE_KEYS = {
    "deepsets": {
        "phi_layers",
        "rho_layers",
        "activation",
        "aggregation",
        "dropout",
        "mask",
    },
    "mlp": {"hidden_layers", "activation", "batchnorm"},
    "jetformer": {
        "embed_dim",
        "num_heads",
        "num_transformers",
        "normalization",
        "dropout",
    },
    "lorentznet": {"n_hidden", "n_layers", "c_weight", "dropout"},
    "pelican": {
        "n_hidden",
        "n_message",
        "n_blocks",
        "dropout",
        "nbar",
        "add_beams",
        "n_chunks",
    },
    "lgatr_slim": {
        "num_blocks",
        "hidden_v_channels",
        "hidden_s_channels",
        "num_heads",
        "mlp_ratio",
        "dropout",
        "units",
        "add_beams",
        "add_time",
    },
    "lloca_lorentz": {
        "frames",
        "num_blocks",
        "num_heads",
        "attn_scalars",
        "attn_vectors",
        "mlp_factor",
        "dropout",
        "framesnet_hidden",
        "mass_reg",
        "units",
    },
    "vn": {
        "head_layers",
        "num_blocks",
        "vector_channels",
        "scalar_channels",
        "invariants",
        "tensor_invariants",
        "stream_channels",
        "dropout",
        "units",
    },
    "jedi_linear": {"hidden", "head_layers", "activation", "aggregation"},
    "linformer": {
        "embed_dim",
        "kv_proj_dim",
        "ff_dim",
        "head_layers",
        "aggregation",
        "batchnorm",
    },
}
# The same network with the beams switched off, so the same keys.
ARCHITECTURE_KEYS["pelican_lorentz"] = ARCHITECTURE_KEYS["pelican"]
# The same network with the spurion tokens switched off, so the same keys.
ARCHITECTURE_KEYS["lgatr_slim_lorentz"] = ARCHITECTURE_KEYS["lgatr_slim"]
# The same network with another frames class, so the same keys.
ARCHITECTURE_KEYS["lloca_z"] = ARCHITECTURE_KEYS["lloca_lorentz"]
ARCHITECTURE_KEYS["lloca_so2"] = ARCHITECTURE_KEYS["lloca_lorentz"]
ARCHITECTURE_KEYS["lloca_so3"] = ARCHITECTURE_KEYS["lloca_lorentz"]
PT_ETA_PHI = ["pt", "eta", "phi"]
# The rows whose required_features are not the four-momentum, which their experiments
# set as data.features.
READS = {"vn": PT_ETA_PHI}


def _algorithm(
    name: str,
    nconstituents: int = 8,
    method: str = "augmentation",
    group: str = "none",
    features: list[str] = FOUR_MOMENTUM,
    **overrides: object,
) -> JetTaggingModule:
    # The algorithm configs interpolate ${data.*}, so a data node of our own stands in.
    cfg = OmegaConf.create(
        {
            "data": {
                "nconstituents": nconstituents,
                "method": method,
                "augmentation_scheme": group,
                "transform_kwargs": {"lorentz_eta_max": 2.0},
                "features": features,
            }
        }
    )
    cfg.algorithm = OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml")
    cfg.algorithm.merge_with(overrides)
    return instantiate(cfg.algorithm)


def _attach(module: JetTaggingModule, datamodule: StubDataModule) -> JetTaggingModule:
    # setup() reads self._trainer.datamodule; the public trainer property would raise here.
    module._trainer = SimpleNamespace(datamodule=datamodule)
    module.setup("fit")
    return module


# The baseline-only models are never trained under the SCL: the embedded ones carry
# their symmetry in the architecture, and JEDI-linear is a comparator from the
# literature rather than a subject of the study.
@pytest.mark.parametrize(
    "name, method, group",
    [(name, "augmentation", "none") for name in MODELS]
    + [(name, "symm_loss", "so2") for name in MODELS if name not in BASELINE_ONLY],
)
def test_every_algorithm_takes_one_step_under_either_method(
    name: str, method: str, group: str
) -> None:
    features = READS.get(name, FOUR_MOMENTUM)
    module = _algorithm(name, method=method, group=group)
    _attach(module, StubDataModule(features))
    width = len(RECORD_COLUMNS) if method == "symm_loss" else len(features)
    # The momenta stand in for narrower columns: the network sees a width, not a meaning.
    batch = jets(8, 8, 6, max(width, len(FOUR_MOMENTUM)))[..., :width]
    out = module.model_step((batch, labels(8)))

    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (8, 5)
    assert out["preds"].shape == (8,)
    assert ("symm_loss" in out) == (method == "symm_loss")


@pytest.mark.parametrize("name", MODELS)
def test_every_algorithm_records_hparams_and_reduces_a_full_width_batch(
    name: str,
) -> None:
    module = _algorithm(name)
    features = READS.get(name, FOUR_MOMENTUM)

    assert module.hparams.name == name
    assert module.hparams.nbr_feats == len(features)
    assert ARCHITECTURE_KEYS[name] <= set(module.hparams)
    assert module.flops_total > 0

    # Under keep_full_features_at_test the probe's batches are 16 columns wide.
    _attach(module, StubDataModule(features, keep_full_features_at_test=True))
    assert module(jets(4, 8, 6, len(RECORD_COLUMNS))).shape == (4, 5)


# The embedded models read the four-momenta and nothing else, so neither takes a feature
# list of its own.
@pytest.mark.parametrize("name", [name for name in MODELS if name not in EMBEDDED])
def test_a_reduced_feature_list_is_read_at_either_batch_width(name: str) -> None:
    features = ["pt", "eta", "phi"]
    module = _algorithm(name, features=features)
    datamodule = StubDataModule(features, keep_full_features_at_test=True)
    # The momenta stand in for the three columns: the network sees a width, not a meaning.
    narrow = jets(8, 8, 6)[..., :3]

    assert module.hparams.nbr_feats == 3
    _attach(module, datamodule)

    # pt, eta and phi sit at 5, 7 and 10 of the record's 16 columns.
    assert module.model_feature_indices == [5, 7, 10]
    assert module.model_step((narrow, labels(8)))["logits"].shape == (8, 5)
    assert module(jets(4, 8, 6, len(RECORD_COLUMNS))).shape == (4, 5)

    # The SCL transforms the momenta of a full-width batch and reduces it in forward().
    scl = _algorithm(name, method="symm_loss", group="so2", features=features)
    wide = jets(8, 8, 6, len(RECORD_COLUMNS))
    out = _attach(scl, datamodule).model_step((wide, labels(8)))

    assert "symm_loss" in out


def test_a_reordered_or_incomplete_feature_list_is_refused_by_name() -> None:
    # forward() reduces on a width mismatch alone, so a full-width reorder would go unnoticed.
    reordered = StubDataModule(
        list(reversed(RECORD_COLUMNS)), keep_full_features_at_test=True
    )
    with pytest.raises(ValueError, match="different order"):
        _attach(_algorithm("deepsets"), reordered)

    with pytest.raises(ValueError, match="'e'"):
        _attach(_algorithm("lorentznet"), StubDataModule(["px", "py", "pz"]))


def test_each_schedule_is_handed_what_lightning_cannot_infer() -> None:
    module = _algorithm("jetformer")
    module._trainer = SimpleNamespace(estimated_stepping_batches=137)
    one_cycle = module.configure_optimizers()["lr_scheduler"]
    plateau = _algorithm("deepsets").configure_optimizers()["lr_scheduler"]

    assert isinstance(one_cycle["scheduler"], torch.optim.lr_scheduler.OneCycleLR)
    assert one_cycle["scheduler"].total_steps == 137
    assert one_cycle["interval"] == "step"
    assert isinstance(plateau["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert plateau["monitor"] == "val/acc"


def test_deepsets_mean_pools_over_the_jets_own_constituents() -> None:
    # The mean divides by the multiplicity, so a ninth all-zero slot changes nothing; a slot-count
    # mean, as the reference implementation pooled, would scale every pooled vector by 8/9.
    narrow = _algorithm("deepsets", nconstituents=8, aggregation="mean").eval()
    wide = _algorithm("deepsets", nconstituents=9, aggregation="mean").eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    batch = jets(4, 8, 6)

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 4)], dim=1))

    assert torch.allclose(physical, padded, atol=1e-6)


def test_sum_over_n_divides_the_masked_sum_by_the_slot_count() -> None:
    # The divisor is N for every jet, not the multiplicity of 5 or 6, so the pool differs
    # from the sum by one factor, which the next dense layer absorbs. Row 0 has a hole
    # among its real slots.
    generator = torch.Generator().manual_seed(5)
    h = torch.randn(4, 8, 16, generator=generator, dtype=torch.float64)
    real = torch.zeros(4, 8, 1, dtype=torch.bool)
    real[:, :6] = True
    real[0, 2] = False
    masked = h * real
    net = _algorithm("deepsets", nconstituents=8, aggregation="sum_over_n").net
    by_slots = masked.sum(dim=1) / 8

    assert torch.allclose(_masked_pool(h, real, "sum_over_n"), by_slots, rtol=1e-12)
    assert torch.allclose(net._pool(masked, real.double()), by_slots, rtol=1e-12)


# Every pooling the JEDI-linear search offers, under the config's ReLU, and tanh with the
# maximum: the network still accepts tanh, and under it a padded slot that entered the
# maximum as zero would win every channel whose real values are all negative.
JEDI_CHOICES = [
    ("relu", "sum"),
    ("relu", "max"),
    ("relu", "sum_over_n"),
    ("tanh", "max"),
]


def _jedi_net(nconstituents: int = 16, **overrides: object) -> torch.nn.Module:
    # The network alone, in float64: forward() rounds the batch to float32, and this row
    # has no float64 buffer to promote it again, as the de-normalising rows have.
    torch.manual_seed(0)
    module = _algorithm("jedi_linear", nconstituents=nconstituents, **overrides)
    return module.net.eval().double()


@pytest.mark.parametrize("activation, aggregation", JEDI_CHOICES)
def test_jedi_linear_is_blind_to_the_order_of_the_constituents(
    activation: str, aggregation: str
) -> None:
    net = _jedi_net(activation=activation, aggregation=aggregation)
    batch = jets(8, 16, 11).double()
    generator = torch.Generator().manual_seed(2)
    # A permutation of its own for every jet, which also moves padded slots among the
    # real ones.
    order = torch.rand(8, 16, generator=generator).argsort(dim=1)
    shuffled = batch.gather(1, order.unsqueeze(-1).expand_as(batch))

    with torch.no_grad():
        assert torch.allclose(net(batch), net(shuffled), atol=1e-10)


@pytest.mark.parametrize("activation, aggregation", JEDI_CHOICES)
def test_jedi_linear_ignores_padding(activation: str, aggregation: str) -> None:
    # A padded row embeds to the activation of the first bias, so it is the mask that
    # keeps it out of both pools, whether the row trails the jet or sits among its
    # constituents. The jets fill every slot of the narrow network, so the wide one's
    # single padded slot is the only one: under tanh with the maximum it would win every
    # channel whose real values are all negative if it entered as zero. Unlike the
    # embedded rows, the network reads the features as stored, so there is no
    # normalisation of the batch to undo.
    choices = {"activation": activation, "aggregation": aggregation}
    narrow = _jedi_net(nconstituents=8, **choices)
    wide = _algorithm("jedi_linear", nconstituents=9, **choices).net.eval().double()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    batch = jets(4, 8, 8).double()
    zero_slot = torch.zeros(4, 1, 4, dtype=torch.float64)
    holed = batch.clone()
    holed[:, 3] = 0.0
    compacted = torch.cat([holed[:, [0, 1, 2, 4, 5, 6, 7]], zero_slot], dim=1)

    with torch.no_grad():
        padded = wide(torch.cat([batch, zero_slot], dim=1))
        # sum_over_n divides by the slot count, so the wider network scales both pools by
        # 8/9 by construction; the hole keeps N fixed and holds for every pool.
        if aggregation != "sum_over_n":
            assert torch.allclose(narrow(batch), padded, atol=1e-10)
        assert torch.allclose(narrow(holed), narrow(compacted), atol=1e-10)


def test_jedi_linear_sums_over_the_real_constituents_in_both_pools() -> None:
    # At the published point, the sum with ReLU, which the config defaults to.
    # Jet k holds one constituent k times. With the context branch zeroed (it then adds
    # relu(0) = 0), a summed readout makes the pooled vector k times that of one copy;
    # with it live, a summed context reads k times the copy's embedding, so the pooled
    # vector per copy moves with k. A mean readout would fail the first check and a mean
    # context the second: either would hide the multiplicity from the head.
    net = _jedi_net(nconstituents=8)
    constituent = jets(1, 1, 1).double()[0, 0]
    batch = torch.zeros(8, 8, 4, dtype=torch.float64)
    for k in range(1, 9):
        batch[k - 1, :k] = constituent
    copies = torch.arange(1, 9, dtype=torch.float64).unsqueeze(-1)

    with torch.no_grad():
        live = net.pooled(batch) / copies
        for parameter in net.context.parameters():
            parameter.zero_()
        readout = net.pooled(batch) / copies

    assert readout.abs().max() > 0
    assert torch.allclose(readout, readout[:1].expand_as(readout), rtol=1e-12)
    # 5.9 at this initialisation, measured on olgpu on 2026-09-30.
    assert _relative_shift(live[:1], live[-1:]) > 1.0


@pytest.mark.parametrize(
    "features, n_parameters", [(["pt", "eta", "phi"], 19_589), (FOUR_MOMENTUM, 19_653)]
)
def test_jedi_linear_has_the_parameters_of_its_dense_layers_alone(
    features: list[str], n_parameters: int
) -> None:
    # 64 F + 64 for the input layer on F features, 4160 for each of the three 64 x 64
    # layers (local, context, shared) and 4160 + 2080 + 528 + 85 for the head: 64 F +
    # 19 397 at the published point, with none of the released code's BatchNorm, and
    # the same at every multiplicity.
    for nconstituents in (16, 150):
        module = _algorithm(
            "jedi_linear", nconstituents=nconstituents, features=features
        )
        assert sum(p.numel() for p in module.parameters()) == n_parameters


# Every readout pool the search offers, with BatchNorm on and off.
LINFORMER_CHOICES = [
    (aggregation, batchnorm)
    for aggregation in LINFORMER_AGGREGATIONS
    for batchnorm in (True, False)
]


def _linformer_net(nconstituents: int = 8, **overrides: object) -> torch.nn.Module:
    # The network alone, in float64, since forward() rounds the batch to float32, and in
    # eval mode, since BatchNorm in training mode would couple the jets of a batch.
    torch.manual_seed(0)
    module = _algorithm("linformer", nconstituents=nconstituents, **overrides)
    return module.net.eval().double()


def test_linformer_attention_with_identity_projections_is_plain_attention() -> None:
    # With k = N and both projections the identity, each key and value slot is one
    # constituent, so on a jet with every slot real the layer is single-head scaled
    # dot-product attention built from its own dense layers, at the same 1 / sqrt(d).
    net = _linformer_net(nconstituents=8, kv_proj_dim=8)
    attention = net.attention
    real = torch.ones(4, 8, 1, dtype=torch.bool)

    with torch.no_grad():
        attention.e_k.copy_(torch.eye(8))
        attention.e_v.copy_(torch.eye(8))
        h = net.embed(jets(4, 8, 8).double())
        plain = torch.nn.functional.scaled_dot_product_attention(
            attention.q(h), attention.k(h), attention.v(h)
        )
        assert torch.allclose(attention(h, real), attention.out(plain), atol=1e-10)


@pytest.mark.parametrize("aggregation, batchnorm", LINFORMER_CHOICES)
def test_linformer_reads_nothing_through_the_projection_rows_of_its_padded_slots(
    aggregation: str, batchnorm: bool
) -> None:
    # Each slot owns a row of E_k and one of E_v. A padded slot embeds to a function of
    # the first bias, not to zero, so only the mask keeps its rows out of the keys and
    # the values. The last check guards against a vacuous pass: a real slot's row
    # moves the logits.
    net = _linformer_net(aggregation=aggregation, batchnorm=batchnorm)
    batch = jets(8, 8, 6).double()
    generator = torch.Generator().manual_seed(3)

    with torch.no_grad():
        logits = net(batch)
        for projection in (net.attention.e_k, net.attention.e_v):
            projection[6:] = torch.randn(
                2, projection.shape[1], generator=generator, dtype=torch.float64
            )
        assert torch.allclose(net(batch), logits, atol=1e-10)
        net.attention.e_v[0] += 1.0
        assert not torch.allclose(net(batch), logits, atol=1e-6)


def test_the_linformer_pool_reads_the_real_constituents_alone() -> None:
    # After the block a padded token is no longer zero, so the pool must mask it:
    # flooded with 1e6, it would win every channel of the maximum and swamp either sum.
    # Row 0 has a hole among its real slots.
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randn(4, 8, 16, generator=generator, dtype=torch.float64)
    real = torch.zeros(4, 8, 1, dtype=torch.bool)
    real[:, :6] = True
    real[0, 2] = False
    flooded = torch.where(real, tokens, 1e6)

    for aggregation in LINFORMER_AGGREGATIONS:
        pooled = _pool(tokens, real, aggregation)
        assert torch.allclose(_pool(flooded, real, aggregation), pooled), aggregation
    # The divisor is the slot count, 8 for every jet, not the multiplicity of 5 or 6,
    # so the two sums differ by one factor, which the head's first layer absorbs.
    by_slots = _pool(tokens, real, "sum") / 8
    assert torch.allclose(_pool(tokens, real, "sum_over_n"), by_slots, rtol=1e-12)


@pytest.mark.parametrize(
    "features, nconstituents",
    [(FOUR_MOMENTUM, 8), (["pt", "eta", "phi"], 16), (FOUR_MOMENTUM, 150)],
)
def test_linformer_has_the_parameters_of_its_layers_and_two_slot_projections(
    features: list[str], nconstituents: int
) -> None:
    # Every BatchNorm carries a weight and a bias per channel; its running statistics
    # are buffers. At the defaults (d = 16, k = 4, ff = 16, head [16]) the count is
    # 16 F + 8 N + 2085: 2213 at F = 4 and N = 8, 2261 at F = 3 and N = 16.
    module = _algorithm("linformer", nconstituents=nconstituents, features=features)
    hp = module.hparams
    d, k, ff, n_features = hp.embed_dim, hp.kv_proj_dim, hp.ff_dim, len(features)
    widths = [d, *hp.head_layers]
    assert hp.batchnorm

    embedding = n_features * d + d + 2 * d
    # Four d x d dense layers, the key one without a bias, and the (N, k) E_k and E_v.
    attention = 4 * d * d + 3 * d + 2 * nconstituents * k
    feed_forward = d * ff + ff + 2 * ff + ff * d + d
    hidden = sum(a * b + b + 2 * b for a, b in zip(widths, widths[1:]))
    head = hidden + widths[-1] * hp.output_dim + hp.output_dim
    total = embedding + attention + feed_forward + head

    assert total == 16 * n_features + 8 * nconstituents + 2085
    assert sum(p.numel() for p in module.parameters()) == total


def test_linformer_refuses_an_unknown_aggregation_by_name() -> None:
    # The network itself: instantiate() would wrap the error into one of Hydra's.
    with pytest.raises(ValueError, match=r"one of \['sum', 'max', 'sum_over_n'\]"):
        LinformerNet(4, 8, 16, 4, 16, [16], 5, aggregation="mean")


def test_lorentznet_ignores_padding_and_undoes_the_normalisation_of_its_batch() -> None:
    narrow = _algorithm("lorentznet", nconstituents=8).eval()
    wide = _algorithm("lorentznet", nconstituents=9).eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    batch = jets(4, 8, 6)
    scales = {"px": 2.0, "py": 3.0, "pz": 4.0, "e": 5.0}

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 4)], dim=1))
        narrow.norm_scales = scales
        narrow._on_features_resolved()
        normalised = narrow(
            batch / torch.tensor([scales[name] for name in FOUR_MOMENTUM])
        )

    assert torch.allclose(physical, padded, atol=1e-5)
    assert torch.allclose(physical, normalised, atol=1e-3)


def _with_large_coordinate_steps(module: JetTaggingModule) -> JetTaggingModule:
    # A freshly initialised gate is near zero and so is every coordinate step it takes, which is
    # how the reference implementation's per-component clamp went unnoticed here: it only breaks
    # the equivariance once a gate exceeds it, as every trained gate did. At the config's
    # c_weight of 1e-3 a 100 GeV step moves nothing either; at 1 it is a momentum.
    with torch.no_grad():
        for block in module.net.blocks:
            if block.phi_x is not None:
                block.phi_x[-1].weight.mul_(1e5)
            block.c_weight = 1.0
    return module


@pytest.mark.parametrize("scheme", ["so2", "so3", "lorentz", "lorentz_iso"])
def test_lorentznet_logits_survive_any_group_at_large_coordinate_steps(
    scheme: str,
) -> None:
    # Invariance by construction is LorentzNet's whole claim, and the reference the augmented
    # models are read against. In eval mode: in train mode the node-level batch normalisation
    # sees the padded nodes. double() carries the graph into float64: forward() rounds the batch
    # to float32, but the de-normalisation against the float64 buffer promotes everything after.
    torch.manual_seed(0)
    module = _with_large_coordinate_steps(
        _algorithm("lorentznet", nconstituents=16).eval()
    )
    module = module.double()
    batch = jets(8, 16, 11).double() * 10.0
    transformed = make_transform(scheme, FOUR_MOMENTUM, None, 2.0)(batch)

    with torch.no_grad():
        logits, logits_transformed = module(batch), module(transformed)

    # Relative to the logit scale: an absolute 1e-3 on a near-zero network says nothing. The
    # floor is the float32 that forward() and the transform matrices round to, some 1e-7.
    shift = (logits - logits_transformed).abs().max().item() / logits.abs().max().item()
    assert shift < 1e-4


def _pelican(add_beams: bool = True, nconstituents: int = 16) -> JetTaggingModule:
    # float64 throughout: forward() rounds the batch to float32, but the de-normalisation
    # against the float64 buffer promotes everything after it. Eval mode, so the masked batch
    # normalisation reads its buffers rather than the batch.
    torch.manual_seed(0)
    module = _algorithm("pelican", nconstituents=nconstituents, add_beams=add_beams)
    return module.eval().double()


def _relative_shift(logits: torch.Tensor, transformed: torch.Tensor) -> float:
    # Relative to the logit scale: an absolute 1e-3 on a near-zero network says nothing. The
    # floor is the float32 that forward() and the transform matrices round to, some 1e-7.
    return (logits - transformed).abs().max().item() / logits.abs().max().item()


def _shift_under(module: JetTaggingModule, scheme: str) -> float:
    batch = jets(8, 16, 11).double() * 10.0
    transformed = make_transform(scheme, FOUR_MOMENTUM, None, 2.0)(batch)
    with torch.no_grad():
        return _relative_shift(module(batch), module(transformed))


def test_pelican_ignores_padding_and_undoes_the_normalisation_of_its_batch() -> None:
    narrow = _algorithm("pelican", nconstituents=8).eval()
    wide = _algorithm("pelican", nconstituents=9).eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    batch = jets(4, 8, 6)
    scales = {"px": 2.0, "py": 3.0, "pz": 4.0, "e": 5.0}

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 4)], dim=1))
        narrow.norm_scales = scales
        narrow._on_features_resolved()
        normalised = narrow(
            batch / torch.tensor([scales[name] for name in FOUR_MOMENTUM])
        )

    assert torch.allclose(physical, padded, atol=1e-5)
    assert torch.allclose(physical, normalised, atol=1e-3)


@pytest.mark.parametrize(
    "add_beams, scheme",
    [(True, "so2")] + [(False, scheme) for scheme in GROUPS],
)
def test_pelican_is_invariant_under_the_group_the_beams_leave_alone(
    add_beams: bool, scheme: str
) -> None:
    # The network reads Minkowski products alone, so without the beams it is invariant under
    # the whole Lorentz group. The beams are two fixed null vectors, and the only transforms
    # leaving the set of their products with the constituents alone are the rotations about
    # the beam axis.
    assert _shift_under(_pelican(add_beams=add_beams), scheme) < 1e-4


@pytest.mark.parametrize("scheme", ["so3", "lorentz", "lorentz_iso"])
def test_pelican_beams_break_every_group_that_moves_the_beam_axis(scheme: str) -> None:
    # A rotation off the beam tilts the axis; a boost of rapidity y along it rescales the
    # products E -/+ p_z by e^{-/+ y}, because a beam is a fixed vector and does not come
    # along. Either way the beam channels of the pair tensor move, so PELICAN measures these
    # three groups at test time as the augmented models do.
    # Read against the SO(2) shift of the same network rather than an absolute bar: at
    # initialisation the logits are nearly constant (0.2, varying by 2e-4 from jet to jet,
    # the masked normalisation still holding its initial buffers), so the broken groups shift
    # them by 1e-4 to 1e-3 where SO(2) and the beamless network sit at the 1e-8 float32 floor.
    module = _pelican()
    assert _shift_under(module, scheme) > 1000 * _shift_under(module, "so2")


def test_pelican_is_blind_to_the_sign_of_the_beam_axis() -> None:
    # p_z -> -p_z swaps the two beams, which carry one label between them, and leaves every
    # other product untouched; the network is permutation invariant, so the logits stand.
    module = _pelican()
    batch = jets(8, 16, 11).double() * 10.0
    flipped = batch * torch.tensor([1.0, 1.0, -1.0, 1.0]).double()

    with torch.no_grad():
        assert torch.allclose(module(batch), module(flipped), atol=1e-10)


def test_lgatr_slim_ignores_padding_and_undoes_the_normalisation_of_its_batch() -> None:
    # This is the check of the dense key-padding mask.
    narrow = _algorithm("lgatr_slim", nconstituents=8).eval()
    wide = _algorithm("lgatr_slim", nconstituents=9).eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    batch = jets(4, 8, 6)
    scales = {"px": 2.0, "py": 3.0, "pz": 4.0, "e": 5.0}

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 4)], dim=1))
        narrow.norm_scales = scales
        narrow._on_features_resolved()
        normalised = narrow(
            batch / torch.tensor([scales[name] for name in FOUR_MOMENTUM])
        )

    assert torch.allclose(physical, padded, atol=1e-5)
    assert torch.allclose(physical, normalised, atol=1e-3)


def _lgatr(
    add_beams: bool, add_time: bool, nconstituents: int = 16
) -> JetTaggingModule:
    # float64 throughout: forward() rounds the batch to float32, but the
    # de-normalisation against the float64 buffer promotes everything after it.
    torch.manual_seed(0)
    module = _algorithm(
        "lgatr_slim",
        nconstituents=nconstituents,
        add_beams=add_beams,
        add_time=add_time,
    )
    return module.eval().double()


@pytest.mark.parametrize(
    "add_beams, add_time, scheme",
    [(False, False, scheme) for scheme in GROUPS]
    + [
        (True, True, "so2"),
        (True, False, "so2"),
        (False, True, "so2"),
        (False, True, "so3"),
    ],
)
def test_lgatr_slim_is_invariant_under_the_group_the_spurions_leave_alone(
    add_beams: bool, add_time: bool, scheme: str
) -> None:
    # Every layer is Lorentz-equivariant and the logits are scalars, so without spurions
    # the whole Lorentz group holds. The spurions are fixed four-vectors that the
    # transforms leave untouched: the beams, two null vectors along the axis, keep only
    # the rotations about it, and the time vector alone keeps every rotation.
    assert _shift_under(_lgatr(add_beams, add_time), scheme) < 1e-4


@pytest.mark.parametrize("scheme", ["so3", "lorentz", "lorentz_iso"])
def test_lgatr_slim_spurions_break_every_group_that_moves_them(scheme: str) -> None:
    # A rotation off the beam tilts the axis; a boost of rapidity y along it rescales
    # the products E -/+ p_z of each constituent with the beams by e^{-/+ y} and moves
    # its product E with the time vector. Either way the products of the constituents
    # with the spurion tokens move, so L-GATr-slim measures these three groups at test
    # time as the augmented models do.
    # Read against the SO(2) shift of the same network rather than an absolute bar: at
    # initialisation the broken groups shift the logits by only 1e-5 to 1e-3, some of it
    # below the 1e-4 the exact groups are held to, while SO(2) sits at the float32 floor
    # of the cast in forward(), about 1e-7.
    module = _lgatr(True, True)
    assert _shift_under(module, scheme) > 1000 * _shift_under(module, "so2")


def test_lgatr_slim_is_blind_to_the_sign_of_the_beam_axis() -> None:
    # p_z -> -p_z swaps the products with the two beam tokens, which carry the same
    # scalar, and leaves every other Minkowski product untouched; attention is
    # permutation invariant, so the logits stand.
    module = _lgatr(True, True)
    batch = jets(8, 16, 11).double() * 10.0
    flipped = batch * torch.tensor([1.0, 1.0, -1.0, 1.0]).double()

    with torch.no_grad():
        assert torch.allclose(module(batch), module(flipped), atol=1e-10)


def test_compiling_keeps_the_checkpoint_keys_of_the_eager_network() -> None:
    # A torch.compile wrapper would prefix every key with _orig_mod., and eval.py, which never
    # compiles, could not load what a compiled training saved. Compiling is lazy, so no graph is
    # built here.
    eager = _algorithm("lgatr_slim")
    compiled = _attach(_algorithm("lgatr_slim", compile=True), StubDataModule())
    compiled.on_train_start()

    assert compiled.net._compiled_call_impl is not None
    assert list(compiled.state_dict()) == list(eager.state_dict())


def test_lloca_ignores_padding_and_undoes_the_normalisation_of_its_batch() -> None:
    # This is the check of the packed frames-net call and the key-padding mask
    # together: the edge convolution of the frames-net has no mask of its own, so a
    # padded slot that reached it would move the frame of every real constituent.
    narrow = _algorithm("lloca_lorentz", nconstituents=8).eval()
    wide = _algorithm("lloca_lorentz", nconstituents=9).eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    batch = jets(4, 8, 6)
    scales = {"px": 2.0, "py": 3.0, "pz": 4.0, "e": 5.0}

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 4)], dim=1))
        narrow.norm_scales = scales
        narrow._on_features_resolved()
        normalised = narrow(
            batch / torch.tensor([scales[name] for name in FOUR_MOMENTUM])
        )

    assert torch.allclose(physical, padded, atol=1e-5)
    assert torch.allclose(physical, normalised, atol=1e-3)


def _lloca(frames: str, nconstituents: int = 16) -> JetTaggingModule:
    # float64 throughout: forward() rounds the batch to float32, but the
    # de-normalisation against the float64 buffer promotes everything after it.
    torch.manual_seed(0)
    module = _algorithm("lloca_lorentz", nconstituents=nconstituents, frames=frames)
    return module.eval().double()


# The probe groups each frames class makes exact. The polar-decomposition frames of the
# lorentz row co-transform under the whole group. The z frames boost along the beam and
# rotate about it, so they fix its axis, which a rotation off it or a boost across it
# moves. The so2 frames fix the axis and the time direction, which any boost moves, and
# the so3 frames the time direction alone.
LLOCA_EXACT = {
    "lorentz": GROUPS,
    "z": ("so2", "lorentz"),
    "so2": ("so2",),
    "so3": ("so2", "so3"),
}


@pytest.mark.parametrize(
    "frames, scheme",
    [(frames, scheme) for frames, exact in LLOCA_EXACT.items() for scheme in exact],
)
def test_lloca_is_invariant_under_the_group_its_frames_respect(
    frames: str, scheme: str
) -> None:
    # Each constituent is read in its own frame and the attention carries its messages
    # from frame to frame as tensors, so the logits hold under whatever the frames
    # co-transform with. Every test constituent carries 20 GeV of mass, far above the
    # mass floor of the frames-net, so its lift, which does not commute with a boost,
    # never fires here.
    assert _shift_under(_lloca(frames), scheme) < 1e-4


@pytest.mark.parametrize(
    "frames, scheme",
    [
        (frames, scheme)
        for frames, exact in LLOCA_EXACT.items()
        for scheme in GROUPS
        if scheme not in exact
    ],
)
def test_lloca_frames_break_every_group_they_do_not_respect(
    frames: str, scheme: str
) -> None:
    # The frames hold the beam axis or the time direction fixed, so a transform that
    # moves it is not undone by them: the local momenta the backbone reads move, and
    # LLoCa measures that group at test time as the augmented models do.
    # Read against the SO(2) shift of the same network rather than an absolute bar, as
    # for PELICAN and L-GATr-slim: at initialisation SO(2) sits at the float32 floor of
    # the cast in forward(), about 1e-8, and the broken groups move the logits by 0.1 to
    # 4 times the largest of them.
    module = _lloca(frames)
    assert _shift_under(module, scheme) > 1000 * _shift_under(module, "so2")


def test_lloca_standardises_its_frames_net_on_the_first_training_batch() -> None:
    # A training standardises the edge attributes of the frames-net on its first batch,
    # and eval.py reads the statistics back from the checkpoint. Construction and the
    # eval-mode forwards before the first training batch (the model summary, the sanity
    # check) must leave them unset, and the checkpoint must carry them, or the evaluated
    # network would read its edges on another scale than the trained one did.
    module = _algorithm("lloca_lorentz")
    block = module.net.framesnet.equivectors.block
    # The FLOP count at construction runs in training mode, but restores the buffers.
    assert not block.edge_inited

    module.eval()
    with torch.no_grad():
        module(jets(4, 8, 6))
    assert not block.edge_inited

    module.train()
    with torch.no_grad():
        module(jets(4, 8, 6))
    assert block.edge_inited
    assert block.edge_std.item() != 1.0

    state = module.state_dict()
    for name in ("edge_inited", "edge_mean", "edge_std"):
        assert f"net.framesnet.equivectors.block.{name}" in state


def _with_random_gates(module: torch.nn.Module) -> torch.nn.Module:
    # Every gate starts at exactly 1/2, a constant under which the exactness tests would
    # pass without exercising the gates; N(0, 1) weights make each gate a function of
    # the scalars.
    with torch.no_grad():
        for gate in module.modules():
            if isinstance(gate, SplitGate):
                gate.linear.weight.normal_()
                gate.linear.bias.normal_()
    return module


def _record_jets(n_jets: int = 8, n_slots: int = 16, n_real: int = 11) -> torch.Tensor:
    # The record's 16 columns in float64, pt, eta and phi written from massless momenta:
    # a beam boost then shifts every eta by the same rapidity, and E = |p| holds to
    # 1e-16 rather than to the float32 rounding of E^2. No test reads the other columns.
    momenta = jets(n_jets, n_slots, n_real).double()[..., :3] * 10.0
    px, py, pz = momenta.unbind(dim=-1)
    columns = {
        "px": px,
        "py": py,
        "pz": pz,
        "e": momenta.norm(dim=-1),
        "pt": compute_pt(px, py),
        "eta": compute_eta(px, py, pz),
        "phi": compute_phi(px, py),
    }
    batch = torch.zeros(n_jets, n_slots, len(RECORD_COLUMNS), dtype=torch.float64)
    for name, values in columns.items():
        batch[..., RECORD_COLUMNS.index(f"j1_{name}")] = values
    return batch


def _vn(nconstituents: int = 16) -> JetTaggingModule:
    # float64 throughout: forward() rounds the batch to float32, but the
    # de-normalisation against the float64 buffer promotes everything after it. The
    # datamodule keeps the record's 16 columns, as the probe's batches do, so that
    # forward() reduces them to [pt, eta, phi].
    torch.manual_seed(0)
    module = _algorithm("vn", nconstituents=nconstituents).eval().double()
    datamodule = StubDataModule(PT_ETA_PHI, keep_full_features_at_test=True)
    return _attach(_with_random_gates(module), datamodule)


def _record_shift_under(module: JetTaggingModule, scheme: str) -> float:
    batch = _record_jets()
    transformed = make_transform(scheme, RECORD_COLUMNS, None, 2.0)(batch)
    with torch.no_grad():
        return _relative_shift(module(batch), module(transformed))


# The probe groups the network makes exact: a rotation about the beam shifts every phi,
# and a boost along it every eta of a massless constituent, by the same amount.
VN_EXACT = ("so2", "lorentz")


@pytest.mark.parametrize("scheme", VN_EXACT)
def test_vn_is_invariant_under_the_probe_groups_that_shift_eta_or_phi(
    scheme: str,
) -> None:
    # The angles are measured from the leader, so a common shift never reaches the
    # network. The transform recomputes pt, eta and phi from the moved momenta, and
    # forward() rounds them to float32, which sets the floor: 9e-8 under so2 and 2e-8
    # under the beam boost.
    assert _record_shift_under(_vn(), scheme) < 1e-6


@pytest.mark.parametrize("scheme", [s for s in GROUPS if s not in VN_EXACT])
def test_vn_breaks_every_probe_group_that_moves_the_beam_axis(scheme: str) -> None:
    # A rotation off the beam and a boost across it change every p_T and move the
    # constituents in (eta, phi) by different amounts. Read against the SO(2) shift of
    # the same network, as for the other embedded rows: SO(2) sits at the float32
    # floor, 9e-8, and the broken groups move the logits by 0.5 to 1.8 times the
    # largest of them.
    module = _vn()
    assert _record_shift_under(module, scheme) > 1000 * _record_shift_under(
        module, "so2"
    )


def _vn_net(nconstituents: int = 8) -> VectorNeuronsClassifier:
    # Two blocks, so that the residual path is exercised too; float64, and eval mode so
    # that the batch normalisation reads its buffers.
    torch.manual_seed(0)
    net = VectorNeuronsClassifier(
        nconstituents,
        [16],
        num_blocks=2,
        vector_channels=8,
        scalar_channels=16,
        invariants=8,
        stream_channels=8,
    )
    return _with_random_gates(net.double().eval())


def _wrap(phi: torch.Tensor) -> torch.Tensor:
    return (phi + math.pi) % (2 * math.pi) - math.pi


def _vn_jets(
    n_jets: int = 5,
    n_slots: int = 8,
    n_real: int = 6,
    phi0: float = 0.3,
    shuffled: bool = True,
) -> torch.Tensor:
    # Rows [p_T GeV, eta, phi] of narrow jets about (0.4, phi0) in float64, the padded
    # rows all zero and, when shuffled, among the real ones.
    generator = torch.Generator().manual_seed(1)
    gauss = torch.randn(3, n_jets, n_real, generator=generator, dtype=torch.float64)
    x = torch.zeros(n_jets, n_slots, 3, dtype=torch.float64)
    x[:, :n_real, 0] = 40.0 * gauss[0].exp()
    x[:, :n_real, 1] = 0.4 + 0.3 * gauss[1]
    x[:, :n_real, 2] = _wrap(phi0 + 0.3 * gauss[2])
    if not shuffled:
        return x
    return x[:, torch.randperm(n_slots, generator=generator)]


def _with_angles(x: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    # New angles for the real rows alone: a padded row stays all zero.
    real = (x[..., 0] != 0.0).to(x.dtype)
    return torch.stack([x[..., 0], real * eta, real * phi], dim=-1)


@pytest.mark.parametrize(
    "deta, dphi", [(0.0, 0.7), (0.0, 3.0), (0.0, -2.5), (1.3, 0.0), (-2.0, 1.1)]
)
def test_vn_is_invariant_under_shifts_of_eta_and_of_phi_across_the_wrap(
    deta: float, dphi: float
) -> None:
    # Jets about phi = 2.9 straddle the wrap at pi before or after the shift; the angles
    # measured from the leader are wrapped once more, so that the jet stays whole.
    net, x = _vn_net(), _vn_jets(phi0=2.9)
    shifted = _with_angles(x, x[..., 1] + deta, _wrap(x[..., 2] + dphi))

    with torch.no_grad():
        assert _relative_shift(net(x), net(shifted)) < 1e-9


@pytest.mark.parametrize("centre", [0, 3])
def test_vn_is_invariant_under_rotations_of_the_eta_phi_plane(centre: int) -> None:
    # A rotation of the plane rotates the (eta, phi) components of every lift and fixes
    # the other two, and G is a multiple of the identity on (eta, phi). About the
    # fourth-hardest constituent (centre 3) it also moves the leader, whose subtraction
    # turns it into the same rotation about the leader.
    net, x = _vn_net(), _vn_jets()
    order = x[..., 0].argsort(dim=1, descending=True)
    pivot = x.gather(1, order[:, centre, None, None].expand(-1, 1, 3))
    d_eta, d_phi = x[..., 1] - pivot[..., 1], _wrap(x[..., 2] - pivot[..., 2])
    cos, sin = math.cos(0.9), math.sin(0.9)
    eta = pivot[..., 1] + cos * d_eta - sin * d_phi
    phi = _wrap(pivot[..., 2] + sin * d_eta + cos * d_phi)

    with torch.no_grad():
        assert _relative_shift(net(x), net(_with_angles(x, eta, phi))) < 1e-9


@pytest.mark.parametrize("signs", [(-1.0, 1.0), (1.0, -1.0)])
def test_vn_is_blind_to_the_reflections_of_eta_and_of_phi(
    signs: tuple[float, float],
) -> None:
    # A reflection flips one of the (eta, phi) components of every lift, which G weighs
    # squared.
    net, x = _vn_net(), _vn_jets()
    flipped = _with_angles(x, signs[0] * x[..., 1], signs[1] * x[..., 2])

    with torch.no_grad():
        assert _relative_shift(net(x), net(flipped)) < 1e-9


def test_vn_is_blind_to_the_order_of_the_constituents() -> None:
    net, x = _vn_net(), _vn_jets()
    generator = torch.Generator().manual_seed(2)
    # A permutation of its own for every jet.
    order = torch.rand(5, 8, generator=generator).argsort(dim=1)
    shuffled = x.gather(1, order.unsqueeze(-1).expand_as(x))

    with torch.no_grad():
        assert _relative_shift(net(x), net(shuffled)) < 1e-12


def _widened(
    narrow: dict[str, torch.Tensor], wide: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    # From eight slots to nine. The ranks are the input channels of the first block and
    # of the tensor invariants, and their scalars, two per rank, lead the inputs of the
    # first scalar MLP; an all-zero row leaves all three at zero, its batch
    # normalisation, at initialisation, mapping zero to zero. The eight ranks keep their
    # places and the tail moves up by one rank. The context n / N falls by 8/9, so the
    # two batch-normalisation weights that read it, against running means of zero,
    # rise by 9/8.
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


def test_vn_ignores_padding_and_undoes_the_normalisation_of_its_batch() -> None:
    narrow = _with_random_gates(_algorithm("vn", nconstituents=8))
    wide = _algorithm("vn", nconstituents=9)
    wide.load_state_dict(_widened(narrow.state_dict(), wide.state_dict()), strict=True)
    narrow.eval()
    wide.eval()
    batch = _vn_jets(4, 8, 6, shuffled=False).float()
    holed = batch.clone()
    holed[:, 3] = 0.0
    compacted = torch.cat(
        [holed[:, [0, 1, 2, 4, 5, 6, 7]], torch.zeros(4, 1, 3)], dim=1
    )
    scales = {"pt": 2.0, "eta": 3.0, "phi": 4.0}

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 3)], dim=1))
        assert torch.allclose(narrow(holed), narrow(compacted), atol=1e-6)
        narrow.norm_scales = scales
        narrow._on_features_resolved()
        normalised = narrow(batch / torch.tensor([scales[name] for name in PT_ETA_PHI]))

    assert torch.allclose(physical, padded, atol=1e-5)
    assert torch.allclose(physical, normalised, atol=1e-3)


def test_vn_form_is_the_squared_distance_weighted_by_both_momenta() -> None:
    p = torch.tensor([[3.0, 0.5]], dtype=torch.float64)
    eta = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    phi = torch.tensor([[1.0, 1.4]], dtype=torch.float64)
    u = p.unsqueeze(-1) * _unit_lift(eta, phi)
    jet = u.sum(dim=1)
    metric = _form_matrix().double()

    # dR^2 = 0.3^2 + 0.4^2 = 0.25 and p_1 p_2 = 1.5; B(U, U) = 2 B(u_1, u_2).
    expected = torch.tensor([0.375], dtype=torch.float64)
    assert torch.allclose(_form(u[:, 0], u[:, 1], metric), expected, rtol=1e-12)
    assert torch.allclose(_form(jet, jet, metric), 2 * expected, rtol=1e-12)
    assert torch.allclose(_form(u, u, metric), torch.zeros_like(p), atol=1e-12)


def test_vn_tensor_invariants_are_their_triple_sums() -> None:
    torch.manual_seed(0)
    n = 5
    module = TensorInvariants(n, 3).double()
    generator = torch.Generator().manual_seed(2)
    p = torch.rand(1, n, generator=generator, dtype=torch.float64) + 0.5
    p[0, -1] = 0.0
    eta, phi = 0.5 * torch.randn(2, 1, n, generator=generator, dtype=torch.float64)
    unit = _unit_lift(eta, phi)
    with torch.no_grad():
        out = module(p.unsqueeze(-1) * unit, unit)[0]
    q = p[0]
    dr2 = (eta[0, :, None] - eta[0]) ** 2 + (phi[0, :, None] - phi[0]) ** 2
    a, b = module.left.weight.detach(), module.right.weight.detach()
    c, d, e = module.weight.detach()

    chain = torch.einsum("ki,kj,kl,i,j,l,ij,jl->k", a, c, b, q, q, q, dr2, dr2)
    pair4 = torch.einsum("ki,kj,i,j,ij->k", c, d, q, q, dr2**2)
    triangle = torch.einsum(
        "ki,kj,kl,i,j,l,ij,jl,li->k", c, d, e, q, q, q, dr2, dr2, dr2
    )
    assert _relative_shift(torch.cat([chain, pair4, triangle]), out) < 1e-12


def test_vn_reads_the_multiplicity_on_its_own() -> None:
    # Removing a constituent also removes its p_T, so the softest one is first scaled
    # down by 1e9, since some of the first block's scalars are linear in it. Scaling it
    # down by a further factor of ten moves the logits by about 1e-10 of their scale,
    # removing it by about 0.15: that is the multiplicity, n / N, which every block and
    # the head read.
    net, x = _vn_net(), _vn_jets()
    rows = torch.arange(x.shape[0])
    softest = x[..., 0].masked_fill(x[..., 0] == 0.0, math.inf).argmin(dim=1)
    soft, softer, removed = x.clone(), x.clone(), x.clone()
    soft[rows, softest, 0] *= 1e-9
    softer[rows, softest, 0] *= 1e-10
    removed[rows, softest] = 0.0

    with torch.no_grad():
        logits = net(soft)
        content = _relative_shift(logits, net(softer))
        assert _relative_shift(logits, net(removed)) > 1000 * content


def test_vn_starts_with_every_gate_at_one_half() -> None:
    torch.manual_seed(0)
    module = _algorithm("vn", num_blocks=2)
    gates = [gate for gate in module.net.modules() if isinstance(gate, SplitGate)]

    assert len(gates) == 2
    for gate in gates:
        s = torch.randn(3, gate.linear.in_features)
        u = torch.randn(3, gate.linear.out_features, 4)
        assert torch.equal(gate(s, u), u / 2)


def test_vn_logits_are_finite_on_the_input_of_the_flop_counter() -> None:
    # p_T as often negative as not, so that the leader is no hardest constituent and no
    # row is padded.
    module = _algorithm("vn").train()
    assert torch.isfinite(module(torch.randn(2, 8, 3))).all()


def _ragged(multiplicities: list[int], n_slots: int = 16) -> torch.Tensor:
    batch = jets(len(multiplicities), n_slots, n_slots)
    for row, multiplicity in enumerate(multiplicities):
        batch[row, multiplicity:] = 0.0
    # A hole inside the leading slots of row 0: a chunk width read off the real count rather
    # than off the mask would cut that row short of its last real constituent.
    batch[0, 3] = 0.0
    return batch


def _chunked_pair(**overrides: object) -> tuple[JetTaggingModule, JetTaggingModule]:
    # float64 throughout, as in _pelican: the two must agree to the summation order alone.
    dense = _algorithm("pelican", nconstituents=16, n_chunks=1, **overrides).double()
    chunked = _algorithm("pelican", nconstituents=16, n_chunks=4, **overrides).double()
    chunked.load_state_dict(dense.state_dict(), strict=True)
    return dense, chunked


@pytest.mark.parametrize("training", [False, True])
def test_pelican_chunks_reproduce_the_unchunked_logits(training: bool) -> None:
    # Dropout off: its masks are drawn at the chunk shapes, so only the rest is comparable.
    dense, chunked = _chunked_pair(dropout=0.0)
    dense.train(training)
    chunked.train(training)
    batch = _ragged([16, 1, 9, 3, 14, 5, 12, 2]).double()

    with torch.no_grad():
        assert torch.allclose(dense(batch), chunked(batch), atol=1e-8)


def test_pelican_chunks_pool_one_set_of_batch_normalisation_statistics() -> None:
    # What catches a _track called once per chunk: the running estimates would then decay
    # four times faster than the statistics the training mode normalises by.
    dense, chunked = _chunked_pair(dropout=0.0)
    dense.train()
    chunked.train()
    batch = _ragged([16, 1, 9, 3, 14, 5, 12, 2]).double()

    with torch.no_grad():
        dense(batch)
        chunked(batch)

    reference = dict(dense.named_buffers())
    for name, value in chunked.named_buffers():
        if "running_" in name:
            assert torch.allclose(reference[name], value, atol=1e-8), name
        if "num_batches_tracked" in name:
            assert torch.equal(reference[name], value), name


def test_pelican_chunking_handles_a_batch_shorter_than_its_chunk_count() -> None:
    # tensor_split leaves empty pieces behind below the chunk count, which the FLOP-counting
    # pass hits at every construction: it runs the network at B = 2, and count_flops swallows
    # an exception into a NaN rather than raising.
    module = _algorithm("pelican", nconstituents=16, n_chunks=4).eval()

    assert module.flops_total > 0
    with torch.no_grad():
        assert module(jets(2, 16, 5)).shape == (2, 5)


def _pair_tensor(
    n_slots: int = 6, n_real: int = 4, channels: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A pair tensor as the network builds one: zero on every padded pair, with a multiplicity
    # that counts the real slots rather than all of them.
    torch.manual_seed(1)
    real = torch.zeros(2, n_slots, dtype=torch.bool)
    real[:, :n_real] = True
    pair_mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(-1)
    values = torch.randn(2, n_slots, n_slots, channels).double()
    return torch.where(pair_mask, values, 0.0), pair_mask, real.sum(-1, True).double()


def test_the_pair_layer_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    layer = LinEq2to2(3, 4, nbar=6.0).double()
    x, pair_mask, nobj = _pair_tensor()
    # Real slots onto real slots and padded onto padded, so the mask is its own image.
    perm = torch.tensor([2, 0, 3, 1, 5, 4])

    with torch.no_grad():
        out = layer(x, pair_mask, nobj)
        permuted = layer(x[:, perm][:, :, perm], pair_mask, nobj)

    assert torch.allclose(out[:, perm][:, :, perm], permuted, atol=1e-10)


def _down_columns(v: torch.Tensor, n_particles: int) -> torch.Tensor:
    # (B, C, N) -> (B, C, N, N) whose entry (i, j) is v_j, constant down a column.
    return v.unsqueeze(2).expand(-1, -1, n_particles, -1)


def _along_rows(v: torch.Tensor, n_particles: int) -> torch.Tensor:
    # (B, C, N) -> (B, C, N, N) whose entry (i, j) is v_i, constant along a row.
    return v.unsqueeze(3).expand(-1, -1, -1, n_particles)


def _stacked_basis(x: torch.Tensor, nobj: torch.Tensor) -> torch.Tensor:
    # The reference's eops_2_to_2 (ops[1] to ops[15]), written out from its own formulas as
    # the (B, C, 15, N, N) stack: the compact layer has to be checked against something
    # other than its own algebra.
    inputs = x.permute(0, 3, 1, 2)
    n_particles = inputs.shape[-1]
    n = nobj.view(-1, 1, 1)
    diag = torch.diagonal(inputs, dim1=-2, dim2=-1)
    sum_diag = diag.sum(2, keepdim=True) / n
    sum_rows, sum_cols = inputs.sum(3) / n, inputs.sum(2) / n
    sum_all = inputs.sum((2, 3)) / nobj.view(-1, 1) ** 2
    ops = [
        inputs,
        inputs.transpose(2, 3),
        torch.diag_embed(diag),
        _down_columns(diag, n_particles),
        _along_rows(diag, n_particles),
        torch.diag_embed(sum_cols),
        _down_columns(sum_cols, n_particles),
        _along_rows(sum_cols, n_particles),
        torch.diag_embed(sum_rows),
        _down_columns(sum_rows, n_particles),
        _along_rows(sum_rows, n_particles),
        sum_diag.unsqueeze(3).expand(-1, -1, n_particles, n_particles),
        torch.diag_embed(sum_diag.expand(-1, -1, n_particles)),
        sum_all[..., None, None].expand(-1, -1, n_particles, n_particles),
        torch.diag_embed(sum_all.unsqueeze(-1).expand(-1, -1, n_particles)),
    ]
    return torch.stack(ops, dim=2)


def test_the_compact_pair_layer_matches_a_stacked_basis() -> None:
    torch.manual_seed(0)
    layer = LinEq2to2(3, 4, nbar=6.0).double()
    x, pair_mask, nobj = _pair_tensor()
    n_particles = x.shape[1]
    eye = torch.eye(n_particles).double().view(1, n_particles, n_particles, 1)

    with torch.no_grad():
        # The layer keeps the ten multiplicity-scaled averages in weight slots 0 to 9 and the
        # array and its diagonal, which carry no exponent, in 10 to 14: the reference's
        # ops[6] to ops[15] and then ops[1] to ops[5], its stack rotated by five.
        alphas = torch.cat([layer.alphas, torch.zeros(3, 5).double()], dim=1)
        scale = (nobj.view(-1, 1, 1, 1, 1) / 6.0) ** alphas.view(1, 3, 15, 1, 1)
        ops = _stacked_basis(x, nobj).roll(-5, dims=2) * scale
        out = torch.einsum("cbs,ncbij->nijs", layer.weight, ops) + layer.bias
        # Both biases go in before the activation, the diagonal one on the diagonal alone.
        out = torch.nn.functional.leaky_relu(out + eye * layer.diag_bias, 0.01)

        assert torch.allclose(
            torch.where(pair_mask, out, 0.0), layer(x, pair_mask, nobj), atol=1e-10
        )
