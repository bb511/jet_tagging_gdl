from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.algorithms import JetTaggingModule
from src.algorithms.components.lorentznet import psi
from src.algorithms.components.pelican import LinEq2to2
from src.algorithms.components.vector_neurons import (
    RANK_BLOCKS,
    SplitGate,
    SplitLinear,
    VectorNeuronsClassifier,
    _rank_invariants,
    _sort_by_pt,
)
from src.data.components.augmentation import _sample_boosts, make_transform
from src.symmetry import GROUPS
from tests.conftest import (
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
    "vn_z": {
        "head_layers",
        "symmetry",
        "num_blocks",
        "vector_channels",
        "scalar_channels",
        "invariants",
        "dropout",
        "units",
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
# The same network with another partition of the four-momentum, so the same keys.
ARCHITECTURE_KEYS["vn_so2"] = ARCHITECTURE_KEYS["vn_z"]


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


# The embedded models carry their symmetry in the architecture, so neither is trained under
# the SCL.
@pytest.mark.parametrize(
    "name, method, group",
    [(name, "augmentation", "none") for name in MODELS]
    + [(name, "symm_loss", "so2") for name in MODELS if name not in EMBEDDED],
)
def test_every_algorithm_takes_one_step_under_either_method(
    name: str, method: str, group: str
) -> None:
    module = _attach(_algorithm(name, method=method, group=group), StubDataModule())
    width = len(RECORD_COLUMNS) if method == "symm_loss" else len(FOUR_MOMENTUM)
    out = module.model_step((jets(8, 8, 6, width), labels(8)))

    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (8, 5)
    assert out["preds"].shape == (8,)
    assert ("symm_loss" in out) == (method == "symm_loss")


@pytest.mark.parametrize("name", MODELS)
def test_every_algorithm_records_hparams_and_reduces_a_full_width_batch(
    name: str,
) -> None:
    module = _algorithm(name)

    assert module.hparams.name == name
    assert module.hparams.nbr_feats == len(FOUR_MOMENTUM)
    assert ARCHITECTURE_KEYS[name] <= set(module.hparams)
    assert module.flops_total > 0

    # Under keep_full_features_at_test the probe's batches are 16 columns wide.
    _attach(module, StubDataModule(keep_full_features_at_test=True))
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
    narrow = _algorithm("deepsets", nconstituents=8).eval()
    wide = _algorithm("deepsets", nconstituents=9).eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    assert narrow.net.aggregation == "mean"
    batch = jets(4, 8, 6)

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 4)], dim=1))

    assert torch.allclose(physical, padded, atol=1e-6)


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


def _with_random_gates(module: JetTaggingModule) -> JetTaggingModule:
    # Every gate starts at exactly 1/2, a constant under which the exactness tests would
    # pass without exercising the gates; N(0, 1) weights make each gate a function of
    # the scalars.
    with torch.no_grad():
        for gate in module.net.modules():
            if isinstance(gate, SplitGate):
                gate.linear.weight.normal_()
                gate.linear.bias.normal_()
    return module


def _vn(symmetry: str, nconstituents: int = 16) -> JetTaggingModule:
    # float64 throughout: forward() rounds the batch to float32, but the
    # de-normalisation against the float64 buffer promotes everything after it.
    torch.manual_seed(0)
    module = _algorithm("vn_z", nconstituents=nconstituents, symmetry=symmetry)
    return _with_random_gates(module.eval().double())


# The probe groups each partition makes exact. Every one of them keeps the beam axis
# and p_T, by which the constituents are ranked. The z splits keep (E, pz) together,
# whose Minkowski product a boost along the beam preserves; so2 weighs E and pz apart,
# and any boost mixes them.
VN_EXACT = {"z": ("so2", "lorentz"), "so2": ("so2",)}


@pytest.mark.parametrize(
    "symmetry, scheme",
    [(symmetry, scheme) for symmetry, exact in VN_EXACT.items() for scheme in exact],
)
def test_vn_is_invariant_under_the_group_its_splits_respect(
    symmetry: str, scheme: str
) -> None:
    # Every layer acts on each split with weights of its own, and the only other
    # operations on a vector are the Minkowski products and the gates on whole splits,
    # so a transform that acts within the splits and keeps the metric on each commutes
    # with the network; the rank invariants are products within unions of splits.
    assert _shift_under(_vn(symmetry), scheme) < 1e-4


@pytest.mark.parametrize(
    "symmetry, scheme",
    [
        (symmetry, scheme)
        for symmetry, exact in VN_EXACT.items()
        for scheme in GROUPS
        if scheme not in exact
    ],
)
def test_vn_splits_break_every_group_that_mixes_them(
    symmetry: str, scheme: str
) -> None:
    # A rotation off the beam and a boost off its axis mix pz into the transverse split
    # and change p_T, and with it the ranks; under so2 a boost along the beam mixes E
    # and pz, which carry weights of their own. Read against the SO(2) shift of the same
    # network, as for PELICAN, L-GATr-slim and LLoCa: at initialisation SO(2) sits at
    # the float32 floor of the cast in forward(), below 1e-7, and the broken groups
    # move the logits by 0.45 to 0.76 times the largest of them.
    module = _vn(symmetry)
    assert _shift_under(module, scheme) > 1000 * _shift_under(module, "so2")


@pytest.mark.parametrize("symmetry", VN_EXACT)
def test_vn_is_blind_to_the_reflections_its_splits_keep(symmetry: str) -> None:
    # py -> -py and pz -> -pz act within a split and keep its metric, and neither moves
    # p_T: the flipped component changes sign in every vector and nowhere else, so the
    # logits stand.
    module = _vn(symmetry)
    batch = jets(8, 16, 11).double() * 10.0

    with torch.no_grad():
        logits = module(batch)
        for flip in ([1.0, -1.0, 1.0, 1.0], [1.0, 1.0, -1.0, 1.0]):
            flipped = batch * torch.tensor(flip).double()
            assert torch.allclose(logits, module(flipped), atol=1e-10)


@pytest.mark.parametrize("symmetry", VN_EXACT)
def test_vn_ignores_padding_and_undoes_the_normalisation_of_its_batch(
    symmetry: str,
) -> None:
    # The ranks are the input channels of the first block and index its rank
    # invariants, so a ninth slot adds a column to its linear map and a group of inputs
    # to its MLP, both of which an all-zero row leaves at exactly zero; a hole among
    # the slots is sorted to the end, and the multiplicity is read off the mask.
    narrow = _with_random_gates(_algorithm("vn_z", nconstituents=8, symmetry=symmetry))
    wide = _algorithm("vn_z", nconstituents=9, symmetry=symmetry)
    # The eight ranks' inputs keep their columns, the ninth's are left alone and the
    # tail (the jet entries, the invariants and the context) moves up by one rank.
    per_rank = 2 * len(RANK_BLOCKS[symmetry]) + 1
    state = wide.state_dict()
    for name, value in narrow.state_dict().items():
        if name == "net.blocks.0.mlp.0.weight":
            state[name][:, : 8 * per_rank] = value[:, : 8 * per_rank]
            state[name][:, 9 * per_rank :] = value[:, 8 * per_rank :]
        else:
            state[name][..., : value.shape[-1]] = value
    wide.load_state_dict(state, strict=True)
    narrow.eval()
    wide.eval()
    batch = jets(4, 8, 6)
    holed = batch.clone()
    holed[:, 3] = 0.0
    compacted = torch.cat(
        [holed[:, [0, 1, 2, 4, 5, 6, 7]], torch.zeros(4, 1, 4)], dim=1
    )
    scales = {"px": 2.0, "py": 3.0, "pz": 4.0, "e": 5.0}

    with torch.no_grad():
        physical = narrow(batch)
        padded = wide(torch.cat([batch, torch.zeros(4, 1, 4)], dim=1))
        assert torch.allclose(narrow(holed), narrow(compacted), atol=1e-6)
        narrow.norm_scales = scales
        narrow._on_features_resolved()
        normalised = narrow(
            batch / torch.tensor([scales[name] for name in FOUR_MOMENTUM])
        )

    assert torch.allclose(physical, padded, atol=1e-5)
    assert torch.allclose(physical, normalised, atol=1e-3)


@pytest.mark.parametrize("symmetry", VN_EXACT)
def test_vn_is_blind_to_the_order_of_the_constituents(symmetry: str) -> None:
    module = _vn(symmetry)
    batch = jets(8, 16, 11).double() * 10.0
    generator = torch.Generator().manual_seed(2)
    # A permutation of its own for every jet.
    order = torch.rand(8, 16, generator=generator).argsort(dim=1)
    shuffled = batch.gather(1, order.unsqueeze(-1).expand_as(batch))

    with torch.no_grad():
        assert torch.allclose(module(batch), module(shuffled), atol=1e-10)


@pytest.mark.parametrize(
    "scheme, commutes", [("so2", True), ("lorentz", True), ("so3", False)]
)
def test_the_vn_ranking_by_pt_commutes_with_the_groups_that_keep_the_beam_axis(
    scheme: str, commutes: bool
) -> None:
    # The claim the design rests on: a network that reads the constituents by rank is
    # invariant only under a group that leaves the ranks alone. A rotation about the
    # beam and a boost along it keep every p_T; a rotation off the beam does not.
    # Stable, so the padded slots, all at p_T = 0, keep their order.
    torch.manual_seed(0)
    batch = jets(8, 16, 11).double() * 10.0
    transformed = make_transform(scheme, FOUR_MOMENTUM, None, 2.0)(batch)

    def ranks(x: torch.Tensor) -> torch.Tensor:
        pt2 = x[..., 0] ** 2 + x[..., 1] ** 2
        return pt2.argsort(dim=1, descending=True, stable=True)

    assert torch.equal(ranks(batch), ranks(transformed)) == commutes


def _in_float64(scheme: str, batch: torch.Tensor) -> torch.Tensor:
    # make_transform draws its matrices in the default dtype, and in float32 a rotation
    # is orthogonal to 1e-7 alone, which would swamp a comparison at 1e-10.
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return make_transform(scheme, FOUR_MOMENTUM, None, 2.0)(batch)
    finally:
        torch.set_default_dtype(default)


def _rank_invariants_of(batch: torch.Tensor, symmetry: str) -> torch.Tensor:
    # From the test columns [px, py, pz, e] in GeV to what the network reads: the ranks
    # of [E, px, py, pz] in units of 20 GeV.
    p = _sort_by_pt(batch[..., [3, 0, 1, 2]] / 20.0)
    return _rank_invariants(p, RANK_BLOCKS[symmetry])


# At 16 slots: 16 ranks of 2 * 2 + 1 entries and 3 jet entries on z, 16 ranks of
# 2 * 4 + 1 and 5 on so2.
VN_RANK_WIDTH = {"z": 83, "so2": 149}


@pytest.mark.parametrize("symmetry", VN_EXACT)
def test_vn_rank_invariants_have_their_width_and_vanish_on_padding(
    symmetry: str,
) -> None:
    invariants = _rank_invariants_of(jets(4, 16, 11), symmetry)
    per_rank = 2 * len(RANK_BLOCKS[symmetry]) + 1
    ranks = invariants[:, : 16 * per_rank].unflatten(1, (16, per_rank))

    assert invariants.shape == (4, VN_RANK_WIDTH[symmetry])
    assert torch.equal(ranks[:, 11:], torch.zeros_like(ranks[:, 11:]))
    assert (ranks[:, :11] != 0.0).any(dim=-1).all()


def test_vn_rank_deficits_are_the_angular_distances_to_the_jet() -> None:
    # The stable form of the deficit against its closed forms on z: p_T p_TJ
    # (1 - cos dphi) on (px, py) and m_T m_TJ (cosh dy - 1) on (E, pz), in units.
    p = _sort_by_pt(jets(4, 16, 11).double()[..., [3, 0, 1, 2]] / 20.0)[:, :11]
    jet = p.sum(dim=1, keepdim=True)
    ranks = _rank_invariants(p, RANK_BLOCKS["z"])[:, : 11 * 5].unflatten(1, (11, 5))
    pt, pt_jet = (x[..., 1:3].norm(dim=-1) for x in (p, jet))
    mt, mt_jet = ((x[..., 0] ** 2 - x[..., 3] ** 2).sqrt() for x in (p, jet))
    phi, phi_jet = (torch.atan2(x[..., 2], x[..., 1]) for x in (p, jet))
    y, y_jet = (torch.atanh(x[..., 3] / x[..., 0]) for x in (p, jet))

    transverse = pt * pt_jet * (1.0 - torch.cos(phi - phi_jet))
    longitudinal = mt * mt_jet * (torch.cosh(y - y_jet) - 1.0)
    assert torch.allclose(ranks[..., 2], psi(transverse), rtol=1e-9, atol=0.0)
    assert torch.allclose(ranks[..., 3], psi(longitudinal), rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("symmetry", VN_EXACT)
def test_vn_rank_invariants_hold_under_the_groups_of_their_row(symmetry: str) -> None:
    # Taken on the ranks of the transformed batch, as the network takes them.
    torch.manual_seed(0)
    batch = jets(8, 16, 11).double() * 10.0
    reference = _rank_invariants_of(batch, symmetry)

    for scheme in VN_EXACT[symmetry]:
        moved = _rank_invariants_of(_in_float64(scheme, batch), symmetry)
        assert torch.allclose(moved, reference, rtol=1e-10, atol=0.0), scheme
    rotated = _rank_invariants_of(_in_float64("so3", batch), symmetry)
    assert _relative_shift(reference, rotated) > 1e-3


# The z split each so2 split copies: E and pz both from (E, pz), (px, py) from itself.
Z_TO_SO2 = [1, 1, 0]


def _z_inputs_in_so2(nconstituents: int, invariants: int) -> torch.Tensor:
    # Where each input of the first z MLP sits among the so2 ones. Per rank, so2 reads
    # the squares on (E,), (pz,), (px, py) and (E, pz), the deficits on the same four
    # blocks and the product with the jet, and z the last two blocks of each group and
    # that product; the jet entries follow the same pattern, and the invariants and the
    # context close both.
    ranks = 9 * torch.arange(nconstituents)[:, None] + torch.tensor([2, 3, 6, 7, 8])
    jet = 9 * nconstituents + torch.tensor([2, 3, 4])
    tail = 9 * nconstituents + 5 + torch.arange(invariants + 1)
    return torch.cat([ranks.flatten(), jet, tail])


def _as_so2(z: JetTaggingModule) -> JetTaggingModule:
    # The so2 network that computes what the z network does: its E and pz splits carry
    # the weights of the (E, pz) split, and its first layer reads the inputs z has with
    # z's weights and the others with zero.
    n, invariants = z.hparams.nconstituents, z.hparams.invariants
    so2 = _algorithm("vn_z", nconstituents=n, symmetry="so2").eval().double()
    source = dict(z.net.named_modules())
    with torch.no_grad():
        for name, target in so2.net.named_modules():
            if isinstance(target, SplitLinear):
                target.weight.copy_(source[name].weight[Z_TO_SO2])
            elif isinstance(target, SplitGate):
                # The gate rows are channel-major: row c * splits + split.
                for key in ("weight", "bias"):
                    rows = getattr(source[name].linear, key).unflatten(0, (-1, 2))
                    getattr(target.linear, key).copy_(rows[:, Z_TO_SO2].flatten(0, 1))
            elif name == "blocks.0.mlp.0":
                target.weight.zero_()
                target.weight[:, _z_inputs_in_so2(n, invariants)] = source[name].weight
                target.bias.copy_(source[name].bias)
            elif isinstance(target, torch.nn.Linear) and "gate" not in name:
                target.load_state_dict(source[name].state_dict())
    return so2


def test_a_vn_z_network_is_an_so2_network_with_tied_longitudinal_weights() -> None:
    # The so2 group is a subgroup of the z one, so the so2 networks include the z ones:
    # a z network is the so2 network whose E and pz splits share their weights and
    # whose first layer ignores the rank invariants of E and pz taken apart.
    z = _vn("z")
    batch = jets(8, 16, 11).double() * 10.0

    with torch.no_grad():
        assert torch.allclose(z(batch), _as_so2(z)(batch), atol=1e-10)


def test_tying_the_vn_z_splits_makes_a_linear_layer_commute_with_every_boost() -> None:
    # One weight for all four components is the Lorentz-equivariant layer of
    # L-GATr-slim: separate weights on (px, py) and (E, pz) are what cut the group down
    # to the beam's.
    torch.manual_seed(0)
    layer = SplitLinear(5, 3, "z").double()
    v = torch.randn(4, 5, 4).double()
    along = _sample_boosts(4, 2.0, isotropic=False, device=None).double()
    across = _sample_boosts(4, 2.0, isotropic=True, device=None).double()

    def commutator(boosts: torch.Tensor) -> float:
        def boost(x: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bij,bcj->bci", boosts, x)

        return (layer(boost(v)) - boost(layer(v))).abs().max().item()

    with torch.no_grad():
        assert commutator(along) < 1e-10
        assert commutator(across) > 1e-3
        layer.weight[0] = layer.weight[1]
        assert commutator(across) < 1e-10


def test_vn_reads_the_multiplicity_on_its_own() -> None:
    # Removing a constituent also removes its momentum, so the softest one is first
    # scaled down by 1e9, since some of its rank invariants are linear in it. Scaling it
    # down by a further factor of ten moves the logits by about 3e-8 of their scale,
    # removing it by about 0.3 per cent: that is the multiplicity, which every block
    # and the head read.
    module = _vn("z")
    batch = jets(8, 16, 11).double() * 10.0
    rows = torch.arange(8)
    softest = (batch[..., 0] ** 2 + batch[..., 1] ** 2)[:, :11].argmin(dim=1)
    soft, softer, removed = batch.clone(), batch.clone(), batch.clone()
    soft[rows, softest] *= 1e-9
    softer[rows, softest] *= 1e-10
    removed[rows, softest] = 0.0

    with torch.no_grad():
        logits = module(soft)
        content = _relative_shift(logits, module(softer))
        assert _relative_shift(logits, module(removed)) > 1000 * content


@pytest.mark.parametrize("symmetry", VN_EXACT)
def test_vn_starts_with_every_gate_at_one_half(symmetry: str) -> None:
    torch.manual_seed(0)
    module = _algorithm("vn_z", symmetry=symmetry)
    gates = [gate for gate in module.net.modules() if isinstance(gate, SplitGate)]

    assert len(gates) == module.hparams.num_blocks
    for gate in gates:
        s = torch.randn(3, gate.linear.in_features)
        u = torch.randn(3, gate.linear.out_features // gate.n_split, 4)
        assert torch.equal(gate(s, u), u / 2)
    # The FLOP counter's input, whose energies are as often negative as not.
    module.train()
    assert torch.isfinite(module(torch.randn(2, 8, 4))).all()


def test_vn_refuses_an_unknown_symmetry_by_name() -> None:
    # The network itself: instantiate() would wrap the error into one of Hydra's.
    with pytest.raises(ValueError, match="symmetry must be one of"):
        VectorNeuronsClassifier(8, [32], symmetry="so3")


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
