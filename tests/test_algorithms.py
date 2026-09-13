from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.algorithms import JetTaggingModule
from src.data.components.augmentation import make_transform
from tests.conftest import (
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
}


def _algorithm(
    name: str, nconstituents: int = 8, method: str = "augmentation", group: str = "none"
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
    return instantiate(cfg.algorithm)


def _attach(module: JetTaggingModule, datamodule: StubDataModule) -> JetTaggingModule:
    # setup() reads self._trainer.datamodule; the public trainer property would raise here.
    module._trainer = SimpleNamespace(datamodule=datamodule)
    module.setup("fit")
    return module


# LorentzNet is the symmetry-embedded class, so it is never trained under the SCL.
@pytest.mark.parametrize(
    "name, method, group",
    [(name, "augmentation", "none") for name in MODELS]
    + [(name, "symm_loss", "so2") for name in MODELS if name != "lorentznet"],
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
