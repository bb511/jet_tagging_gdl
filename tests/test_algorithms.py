"""The four algorithms, built from their shipped configs and stepped once each off a stub
datamodule, so the checks cost no download and no epoch.

What they pin is the contract between ``JetTaggingModule`` and its subclasses: the feature
resolution in ``setup``, the two-branch ``model_step``, the hyperparameters the results table and
the checkpoints read back, and the physics LorentzNet is built for.
"""

from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.algorithms import JetTaggingModule
from src.data.components.augmentation import make_transform
from tests.conftest import ROOT

# The record's 16 columns, in its own order: the width every SCL batch arrives at.
RECORD_COLUMNS = (
    "j1_px j1_py j1_pz j1_e j1_erel j1_pt j1_ptrel j1_eta j1_etarel j1_etarot j1_phi "
    "j1_phirel j1_phirot j1_deltaR j1_costheta j1_costhetarel"
).split()
FOUR_MOMENTUM = ["px", "py", "pz", "e"]
ALGORITHMS = ("deepsets", "mlp", "jetformer", "lorentznet")

# Without these keys, which save_hyperparameters collects through Lightning's frame walk, a
# checkpoint cannot rebuild the network it holds.
ARCHITECTURE_KEYS = {
    "deepsets": {"phi_layers", "rho_layers", "activation", "aggregation", "dropout", "mask"},
    "mlp": {"hidden_layers", "activation", "batchnorm"},
    "jetformer": {"embed_dim", "num_heads", "num_transformers", "normalization", "dropout"},
    "lorentznet": {"n_hidden", "n_layers", "c_weight", "dropout"},
}


class _StubDataModule:
    """The surface ``JetTaggingModule.setup`` reads off a datamodule."""

    features = FOUR_MOMENTUM
    feature_names = RECORD_COLUMNS
    norm_scales = dict.fromkeys(RECORD_COLUMNS, 1.0)

    def __init__(self, keep_full_features_at_test: bool = False) -> None:
        self.keep_full_features_at_test = keep_full_features_at_test


def _algorithm(
    name: str, nconstituents: int = 8, method: str = "augmentation", group: str = "none"
) -> JetTaggingModule:
    """Instantiate ``configs/algorithm/<name>.yaml`` against a data node of our own.

    The algorithm configs interpolate ``${data.*}``, which is how one command-line override
    configures the data and the model together.
    """
    cfg = OmegaConf.create(
        {
            "data": {
                "nconstituents": nconstituents,
                "method": method,
                "augmentation_scheme": group,
                "lorentz_eta_max": 2.0,
                "features": FOUR_MOMENTUM,
            }
        }
    )
    cfg.algorithm = OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml")

    return instantiate(cfg.algorithm)


def _attach(module: JetTaggingModule, datamodule: _StubDataModule) -> JetTaggingModule:
    """Run ``setup`` as a trainer would: the base class reads ``self._trainer.datamodule``, so a
    stub is enough, and the public ``trainer`` property, which raises here, is not."""
    module._trainer = SimpleNamespace(datamodule=datamodule)
    module.setup("fit")

    return module


def _jets(n_jets: int, n_slots: int, n_real: int, n_columns: int, seed: int = 0) -> torch.Tensor:
    """Massless constituents of about 20 GeV in the record's first four columns, shape (jets,
    constituents, n_columns), padded slots all-zero."""
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n_jets, n_slots, 3, generator=generator) * 20.0
    batch = torch.zeros(n_jets, n_slots, n_columns)
    batch[..., :3] = momenta
    batch[..., 3] = momenta.norm(dim=-1)
    batch[:, n_real:, :] = 0.0

    return batch


def _labels(n_jets: int, seed: int = 1) -> torch.Tensor:
    """One-hot flavours, as the datamodule serves them."""
    generator = torch.Generator().manual_seed(seed)
    return torch.eye(5)[torch.randint(0, 5, (n_jets,), generator=generator)]


# LorentzNet is the symmetry-embedded comparison class, so it is never trained under the SCL.
@pytest.mark.parametrize(
    "name, method, group",
    [(name, "augmentation", "none") for name in ALGORITHMS]
    + [(name, "symm_loss", "so2") for name in ALGORITHMS if name != "lorentznet"],
)
def test_every_algorithm_takes_one_step_under_either_method(
    name: str, method: str, group: str
) -> None:
    """A finite loss and one prediction per jet is all train.py asks of an algorithm."""
    module = _algorithm(name, method=method, group=group)
    _attach(module, _StubDataModule(keep_full_features_at_test=False))
    width = len(RECORD_COLUMNS) if method == "symm_loss" else len(FOUR_MOMENTUM)
    out = module.model_step((_jets(8, 8, 6, width), _labels(8)))

    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (8, 5)
    assert out["preds"].shape == (8,)
    assert ("symm_loss" in out) == (method == "symm_loss")


@pytest.mark.parametrize("name", ALGORITHMS)
def test_every_algorithm_records_hparams_and_reduces_a_full_width_batch(name: str) -> None:
    """The results table reads flops_total, and a reload rebuilds the net from these keys."""
    module = _algorithm(name)

    assert module.hparams.name == name
    assert module.hparams.nbr_feats == len(FOUR_MOMENTUM)
    assert ARCHITECTURE_KEYS[name] <= set(module.hparams)
    assert module.hparams["flops_total"] > 0

    # Under keep_full_features_at_test the probe's batches are 16 columns wide.
    _attach(module, _StubDataModule(keep_full_features_at_test=True))
    assert module(_jets(4, 8, 6, len(RECORD_COLUMNS))).shape == (4, 5)


def test_a_reordered_or_incomplete_feature_list_is_refused_by_name() -> None:
    """Forward() reduces on a width mismatch alone, so a full-width reorder would go unnoticed;
    LorentzNet on a feature list without the energy is the absence this catches."""
    reordered = _StubDataModule(keep_full_features_at_test=True)
    reordered.features = list(reversed(RECORD_COLUMNS))
    with pytest.raises(ValueError, match="different order"):
        _attach(_algorithm("deepsets"), reordered)

    incomplete = _StubDataModule(keep_full_features_at_test=False)
    incomplete.features = ["px", "py", "pz"]
    with pytest.raises(ValueError, match="'e'"):
        _attach(_algorithm("lorentznet"), incomplete)


def test_each_schedule_is_handed_what_lightning_cannot_infer() -> None:
    """OneCycleLR needs the run length; ReduceLROnPlateau never fires without a monitor key."""
    module = _algorithm("jetformer")
    module._trainer = SimpleNamespace(estimated_stepping_batches=137)
    one_cycle = module.configure_optimizers()["lr_scheduler"]
    plateau = _algorithm("deepsets").configure_optimizers()["lr_scheduler"]

    assert isinstance(one_cycle["scheduler"], torch.optim.lr_scheduler.OneCycleLR)
    assert one_cycle["scheduler"].total_steps == 137
    assert one_cycle["interval"] == "step"
    assert isinstance(plateau["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert plateau["monitor"] == "val/acc"


def test_lorentznet_ignores_padding_and_undoes_the_normalisation_of_its_batch() -> None:
    """Its edges join real constituents only, and its Minkowski invariants are dimensionful."""
    narrow = _algorithm("lorentznet", nconstituents=8).eval()
    wide = _algorithm("lorentznet", nconstituents=9).eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    jets = _jets(4, 8, 6, 4)
    scales = {"px": 2.0, "py": 3.0, "pz": 4.0, "e": 5.0}

    with torch.no_grad():
        physical = narrow(jets)
        padded = wide(torch.cat([jets, torch.zeros(4, 1, 4)], dim=1))
        narrow.norm_scales = scales
        narrow._on_features_resolved()
        normalised = narrow(jets / torch.tensor([scales[name] for name in FOUR_MOMENTUM]))

    assert torch.allclose(physical, padded, atol=1e-5)
    assert torch.allclose(physical, normalised, atol=1e-3)


def test_lorentznet_logits_survive_a_boost() -> None:
    """Invariance by construction is the whole claim, and the reference the others are read by."""
    # In eval mode: in train mode the node-level batch normalisation sees the padded nodes, so the
    # invariance holds only with running statistics.
    torch.manual_seed(0)
    module = _algorithm("lorentznet", nconstituents=16).eval()
    jets = _jets(8, 16, 11, 4)
    boosted, _ = make_transform("lorentz", FOUR_MOMENTUM, None, 2.0)(jets)

    with torch.no_grad():
        shift = (module(jets) - module(boosted)).abs().max().item()

    assert shift < 1e-3
