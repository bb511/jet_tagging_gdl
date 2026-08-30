"""The four algorithms, built from their shipped configs and stepped once each.

Everything here runs off a stub datamodule, so the checks cost no download and no epoch. What they
pin is the contract between ``JetTaggingModule`` and its subclasses: the feature resolution in
``setup``, the two-branch ``model_step``, the hyperparameters the results table and the checkpoints
read back, and the physics LorentzNet is built for.
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

# What save_hyperparameters must have collected through Lightning's frame walk, per
# algorithm: without these a checkpoint cannot rebuild the network it holds.
ARCHITECTURE_KEYS = {
    "deepsets": {"phi_layers", "rho_layers", "activation", "aggregation", "dropout", "mask"},
    "mlp": {"hidden_layers", "activation", "batchnorm"},
    "jetformer": {"embed_dim", "num_heads", "num_transformers", "normalization", "dropout"},
    "lorentznet": {"n_hidden", "n_layers", "c_weight", "dropout"},
}


class _StubDataModule:
    """The surface ``JetTaggingModule.setup`` reads off a datamodule.

    :param keep_full_features_at_test: as the datamodule's own flag; with the SCL it is what
        decides whether the batches carry the record's 16 columns or only four.
    """

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
    """Run ``setup`` as a trainer would, without one.

    The base class reads ``self._trainer.datamodule``, so a stub trainer is enough and the public
    ``trainer`` property, which raises when the module stands alone, is not.
    """
    module._trainer = SimpleNamespace(datamodule=datamodule)
    module.setup("fit")

    return module


def _jets(n_jets: int, n_slots: int, n_real: int, n_columns: int, seed: int = 0) -> torch.Tensor:
    """Massless constituents of about 20 GeV, in the record's first four columns.

    :return: shape (jets, constituents, n_columns), padded slots all-zero.
    """
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


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("method, group", [("augmentation", "none"), ("symm_loss", "so2")])
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
def test_every_algorithm_records_the_hyperparameters_a_checkpoint_needs(name: str) -> None:
    """The results table reads flops_total, and a reload rebuilds the net from these keys."""
    module = _algorithm(name)

    assert module.hparams.name == name
    assert module.hparams.nbr_feats == len(FOUR_MOMENTUM)
    assert ARCHITECTURE_KEYS[name] <= set(module.hparams)
    assert module.hparams["flops_total"] > 0


@pytest.mark.parametrize("name", ALGORITHMS)
def test_a_test_batch_of_the_full_record_width_is_reduced_in_forward(name: str) -> None:
    """Under keep_full_features_at_test the probe's batches are 16 columns wide."""
    module = _algorithm(name)
    _attach(module, _StubDataModule(keep_full_features_at_test=True))

    assert module(_jets(4, 8, 6, len(RECORD_COLUMNS))).shape == (4, 5)


def test_a_reordered_full_width_feature_list_is_refused() -> None:
    """Forward() reduces on a width mismatch alone, so this reorder would go unnoticed."""
    datamodule = _StubDataModule(keep_full_features_at_test=True)
    datamodule.features = list(reversed(RECORD_COLUMNS))
    module = _algorithm("deepsets")

    with pytest.raises(ValueError, match="different order"):
        _attach(module, datamodule)


def test_an_absent_feature_is_named_in_the_error() -> None:
    """LorentzNet on a feature list without the energy is the case this catches."""
    datamodule = _StubDataModule(keep_full_features_at_test=False)
    datamodule.features = ["px", "py", "pz"]
    module = _algorithm("lorentznet")

    with pytest.raises(ValueError, match="'e'"):
        _attach(module, datamodule)


def test_the_one_cycle_schedule_takes_its_step_count_from_the_trainer() -> None:
    """OneCycleLR needs the run length, which is knowable only in configure_optimizers."""
    module = _algorithm("jetformer")
    module._trainer = SimpleNamespace(estimated_stepping_batches=137)

    config = module.configure_optimizers()
    scheduler = config["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    assert scheduler.total_steps == 137
    assert config["lr_scheduler"]["interval"] == "step"


def test_the_plateau_schedule_is_handed_the_metric_it_watches() -> None:
    """ReduceLROnPlateau silently never fires if Lightning is given no monitor key."""
    config = _algorithm("deepsets").configure_optimizers()

    assert isinstance(
        config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    assert config["lr_scheduler"]["monitor"] == "val/acc"


def test_lorentznet_ignores_an_added_padded_constituent() -> None:
    """Its edges are built between real constituents, so a zero row must change nothing."""
    narrow = _algorithm("lorentznet", nconstituents=8).eval()
    wide = _algorithm("lorentznet", nconstituents=9).eval()
    wide.load_state_dict(narrow.state_dict(), strict=True)
    jets = _jets(4, 8, 6, 4)

    with torch.no_grad():
        base = narrow(jets)
        padded = wide(torch.cat([jets, torch.zeros(4, 1, 4)], dim=1))

    assert torch.allclose(base, padded, atol=1e-5)


def test_lorentznet_undoes_the_normalisation_of_its_batch() -> None:
    """Its Minkowski invariants are dimensionful, so a normalised batch is not a rescaling."""
    module = _algorithm("lorentznet", nconstituents=8).eval()
    jets = _jets(4, 8, 6, 4)
    scales = {"px": 2.0, "py": 3.0, "pz": 4.0, "e": 5.0}

    with torch.no_grad():
        physical = module(jets)
        module.norm_scales = scales
        module._on_features_resolved()
        normalised = module(jets / torch.tensor([scales[name] for name in FOUR_MOMENTUM]))

    assert torch.allclose(physical, normalised, atol=1e-3)


@pytest.mark.parametrize("seed", range(4))
def test_lorentznet_logits_survive_a_boost(seed: int) -> None:
    """Invariance by construction is the whole claim, and the reference the others are read
    against.

    Held to 1e-3 in eval mode: in train mode the node-level batch normalisation inside the
    blocks sees the padded nodes, so the invariance holds only with running statistics.
    """
    torch.manual_seed(seed)
    module = _algorithm("lorentznet", nconstituents=16).eval()
    jets = _jets(8, 16, 11, 4, seed=seed)
    boosted, _ = make_transform("lorentz", FOUR_MOMENTUM, None, 2.0)(jets)

    with torch.no_grad():
        shift = (module(jets) - module(boosted)).abs().max().item()

    assert shift < 1e-3
