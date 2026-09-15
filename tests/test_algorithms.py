from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.algorithms import JetTaggingModule
from src.algorithms.components.pelican import LinEq2to2
from src.data.components.augmentation import make_transform
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
}


def _algorithm(
    name: str,
    nconstituents: int = 8,
    method: str = "augmentation",
    group: str = "none",
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
                "features": FOUR_MOMENTUM,
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
        # The reference scales the last ten basis elements alone: ops[1] to ops[5] read the
        # array and its diagonal, whose values do not depend on the multiplicity.
        alphas = torch.cat([torch.zeros(3, 5).double(), layer.alphas], dim=1)
        scale = (nobj.view(-1, 1, 1, 1, 1) / 6.0) ** alphas.view(1, 3, 15, 1, 1)
        ops = _stacked_basis(x, nobj) * scale
        out = torch.einsum("cbs,ncbij->nijs", layer.weight, ops) + layer.bias
        # Both biases go in before the activation, the diagonal one on the diagonal alone.
        out = torch.nn.functional.leaky_relu(out + eye * layer.diag_bias, 0.01)

        assert torch.allclose(
            torch.where(pair_mask, out, 0.0), layer(x, pair_mask, nobj), atol=1e-10
        )
