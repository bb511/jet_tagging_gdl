from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.algorithms.losses.symmetry import SymmetryConsistencyLoss
from src.data import hls4ml_datamodule as datamodule_module
from src.data.components.augmentation import (
    AXIS_COLUMNS,
    _apply,
    compute_eta,
    compute_phi,
    compute_pt,
    make_drawing_transform,
    make_transform,
    rebase_on_kept_axis,
)
from src.data.hls4ml_datamodule import (
    HLS4MLDataModule,
    compose_record_config,
    fold_indices,
)
from src.data.hls4ml_jet_level_datamodule import JetLevelDataModule, shard_paths
from src.symmetry import GROUPS
from tests.conftest import (
    CACHE_DIR,
    JET_LEVEL_COLUMNS,
    JET_LEVEL_SIZES,
    MASS,
    RECORD_COLUMNS,
    ROOT,
    labels,
    needs_record,
    record_jets,
    wrap_phi,
)

SPLIT_SIZES = {"train": 40, "validation": 20, "test": 16}
# Every column at a scale of its own, as the record stores them.
SCALES = {name: 1.5 + 0.5 * i for i, name in enumerate(RECORD_COLUMNS)}
SCALE = torch.tensor(list(SCALES.values()), dtype=torch.float64)
# The record columns a group leaves unchanged and those it moves, |p| standing for the
# norm of the momentum.
ACTIONS = {
    "so2": (("pt", "pz", "eta", "etarel", "phirel"), ("phi",)),
    "so3": (("|p|",), ("pz",)),
    "lorentz": (("pt", "phi", "phirel"), ("eta",)),
    "lorentz_iso": ((), ("eta",)),
}


class FakeRecord:
    # The surface HLS4MLDataModule asks of the record's loader, over jets of a fixed
    # seed whose columns are all random: the rebase must rewrite the axis columns.

    feature_names = RECORD_COLUMNS
    cache_folder = Path("/nowhere/nconst_8")

    def __init__(self) -> None:
        self.prepared = False

    @property
    def norm_params(self) -> dict[str, tuple[str, float]] | None:
        # None until prepare() has run, as the record's own property is.
        if not self.prepared:
            return None
        return {name: ("centered", 1.0 + i) for i, name in enumerate(RECORD_COLUMNS)}

    def prepare(self) -> None:
        self.prepared = True

    def load(self, split: str) -> SimpleNamespace:
        generator = torch.Generator().manual_seed(len(split))
        n_jets = SPLIT_SIZES[split]
        x = torch.randn(n_jets, 8, len(RECORD_COLUMNS), generator=generator)
        return SimpleNamespace(x=x, y=labels(n_jets, seed=len(split)))


@pytest.fixture
def record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeRecord:
    fake = FakeRecord()
    monkeypatch.setattr(datamodule_module, "record_snapshot", lambda *a: str(tmp_path))
    monkeypatch.setattr(
        datamodule_module,
        "compose_record_config",
        lambda *a: SimpleNamespace(data=None),
    )
    monkeypatch.setattr(datamodule_module, "instantiate", lambda *a: fake)
    return fake


def _datamodule(**overrides) -> HLS4MLDataModule:
    defaults = {
        "nconstituents": 8,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
    }
    return HLS4MLDataModule(**(defaults | overrides))


def _axis_columns(x: torch.Tensor, scales: dict[str, float]) -> torch.Tensor:
    # etarel, phirel, deltaR and ptrel from the axis of the real slots, in the stored
    # normalisation, written out from their definitions.
    px, py, pz = (x[..., i] * scales[RECORD_COLUMNS[i]] for i in (0, 1, 2))
    real = (px != 0) | (py != 0)
    jet = [(p * real).sum(dim=1, keepdim=True) for p in (px, py, pz)]
    etarel = compute_eta(px, py, pz) - compute_eta(*jet)
    phirel = wrap_phi(compute_phi(px, py) - compute_phi(*jet[:2]))
    ptrel = compute_pt(px, py) / compute_pt(*jet[:2])
    values = (etarel, phirel, torch.sqrt(etarel**2 + phirel**2), ptrel)
    stored = [v / scales[f"j1_{name}"] for v, name in zip(values, AXIS_COLUMNS)]
    return torch.stack(stored, dim=-1) * real.unsqueeze(-1)


def test_fold_indices_partition_the_pool_by_its_length_and_seed() -> None:
    folds = [fold_indices(23, 5, 7, fold_idx) for fold_idx in range(5)]

    validated = np.concatenate([val_idx for _, val_idx in folds])
    assert sorted(validated) == list(range(23))
    for train_idx, val_idx in folds:
        assert sorted(np.concatenate([train_idx, val_idx])) == list(range(23))
    assert np.array_equal(fold_indices(23, 5, 7, 2)[1], folds[2][1])
    assert not np.array_equal(fold_indices(23, 5, 8, 2)[1], folds[2][1])


def test_the_datamodule_serves_fold_indices_of_the_pooled_splits(
    record: FakeRecord,
) -> None:
    # Training then validation: the order a script must pool them in to select the same
    # jets, as the jet-level datamodule does with the same seed and revision.
    folded = _datamodule(
        n_folds=3, kfold_seed=7, fold_idx=2, features=["pt", "eta", "phi"]
    )
    folded.prepare_data()
    folded.setup()
    single = _datamodule(n_folds=1, fold_idx=4, features=["pt", "eta", "phi"])
    single.setup(stage="fit")
    splits = [folded.load_split(split) for split in ("train", "validation")]
    x_all, y_all = (torch.cat(tensors) for tensors in zip(*splits))
    train_idx, val_idx = fold_indices(len(x_all), 3, 7, 2)
    columns = folded.feature_indices()
    data_configs = [
        OmegaConf.load(ROOT / f"configs/data/{n}.yaml") for n in ("hls4ml", "jet_level")
    ]

    assert columns == [5, 7, 10]
    assert torch.equal(folded.data_train.X, x_all[train_idx][..., columns])
    assert torch.equal(folded.data_val.y, y_all[val_idx])
    assert folded.data_test.n == SPLIT_SIZES["test"]
    # At n_folds 1 the record's own split, whatever fold_idx says, which load_split
    # returns whole: the quantised row traces its training split at test time.
    assert torch.equal(single.load_split("train")[0][..., columns], single.data_train.X)
    assert len({(c.revision, c.kfold_seed) for c in data_configs}) == 1


@pytest.mark.parametrize(
    "method, scheme, fit_width",
    [
        ("augmentation", "none", 4),
        ("augmentation", "so3", 4),
        # The SCL transforms inside model_step, so its fit batches are full width too.
        ("symm_loss", "lorentz", 16),
    ],
)
def test_the_datamodule_serves_rebased_jets_at_the_widths_of_its_regime(
    record: FakeRecord, method: str, scheme: str, fit_width: int
) -> None:
    datamodule = _datamodule(method=method, augmentation_scheme=scheme, batch_size=12)
    datamodule.prepare_data()
    datamodule.setup()
    loaders = {
        "train": datamodule.train_dataloader(),
        "validation": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }
    batches = {split: list(loader) for split, loader in loaders.items()}
    absolute = [0, 1, 2, 3, 5, 7, 10]
    axis = [RECORD_COLUMNS.index(f"j1_{name}") for name in AXIS_COLUMNS]

    assert {x.shape[-1] for x, _ in batches["train"] + batches["validation"]} == {
        fit_width
    }
    assert {x.shape[-1] for x, _ in batches["test"]} == {len(RECORD_COLUMNS)}
    # A batch of 12 divides none of 40, 20 and 16 jets: only the test keeps its tail.
    jets_seen = {
        split: sum(len(y) for _, y in found) for split, found in batches.items()
    }
    assert jets_seen == {"train": 36, "validation": 12, "test": 16}
    assert (datamodule.data_train.transform is not None) == (scheme == "so3")
    assert datamodule.augmentation_scheme == scheme
    stored = FakeRecord().load("test").x
    served = datamodule.data_test.X
    expected = _axis_columns(stored, datamodule.norm_scales)
    assert torch.allclose(served[..., axis], expected, atol=1e-5, rtol=1e-4)
    assert torch.equal(served[..., absolute], stored[..., absolute])


def test_full_axis_keeps_the_record_columns_in_the_baseline_alone(
    record: FakeRecord,
) -> None:
    datamodule = _datamodule(full_axis=True)
    datamodule.prepare_data()
    datamodule.setup()

    assert torch.equal(datamodule.data_test.X, FakeRecord().load("test").x)
    assert torch.equal(datamodule.load_split("train")[0], FakeRecord().load("train").x)
    with pytest.raises(ValueError, match="baseline"):
        _datamodule(full_axis=True, augmentation_scheme="so2")


def test_the_rebase_measures_the_axis_columns_from_the_kept_constituents() -> None:
    # The record measures them from the untruncated jet's axis, which disagrees with the
    # kept one on 70 to 85 % of the entries at 16 constituents; stand-ins of 3 here.
    batch = record_jets(6, 8, 5, mass=MASS) / SCALE
    axis = [RECORD_COLUMNS.index(f"j1_{name}") for name in AXIS_COLUMNS]
    untouched = [i for i in range(len(RECORD_COLUMNS)) if i not in axis]
    batch[:, :5, axis] = 3.0

    rebased = rebase_on_kept_axis(batch, RECORD_COLUMNS, SCALES)

    assert torch.allclose(rebased[..., axis], _axis_columns(batch, SCALES), atol=1e-10)
    assert torch.equal(rebased[..., untouched], batch[..., untouched])
    assert torch.equal(rebase_on_kept_axis(rebased, RECORD_COLUMNS, SCALES), rebased)
    identity = torch.eye(4, dtype=torch.float64).expand(6, 4, 4)
    assert torch.allclose(_apply(rebased, identity, RECORD_COLUMNS, SCALES), rebased)


def _matrices(group: str, draw: torch.Tensor) -> torch.Tensor:
    # The matrix each jet was transformed by, rebuilt from its stored draw: the angle
    # about the beam, the rotation itself, or the rapidity along the beam or along a
    # stored unit axis.
    if group == "so3":
        return draw
    if group == "so2":
        cos, sin, zero, one = draw.cos(), draw.sin(), 0 * draw, 1 + 0 * draw
        rows = [[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]]
        return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
    beam = torch.tensor([0.0, 0.0, 1.0]).expand(len(draw), 3)
    xi, n = (draw, beam) if group == "lorentz" else (draw[:, 0], draw[:, 1:])
    boost = torch.eye(4).repeat(len(xi), 1, 1)
    boost[:, 0, 0] = torch.cosh(xi)
    boost[:, 0, 1:] = boost[:, 1:, 0] = -torch.sinh(xi)[:, None] * n
    boost[:, 1:, 1:] += (
        (torch.cosh(xi) - 1)[:, None, None] * n[:, :, None] * n[:, None, :]
    )
    return boost


def _physical(batch: torch.Tensor, name: str) -> torch.Tensor:
    x = batch[:, :11] * SCALE
    if name == "|p|":
        return x[..., :3].norm(dim=-1)
    return x[..., RECORD_COLUMNS.index(f"j1_{name}")]


@pytest.mark.parametrize("group", GROUPS)
def test_each_group_acts_as_stated_and_replays_from_its_draws(group: str) -> None:
    batch = record_jets(mass=MASS) / SCALE
    torch.manual_seed(0)
    transformed, draw = make_drawing_transform(group, RECORD_COLUMNS, SCALES, 2.0)(
        batch
    )
    torch.manual_seed(0)
    again = make_transform(group, RECORD_COLUMNS, SCALES, 2.0)(batch)
    replayed = _apply(batch, _matrices(group, draw), RECORD_COLUMNS, SCALES)
    unchanged, moved = ACTIONS[group]
    energy = _physical(transformed, "e")
    mass = energy**2 - _physical(transformed, "|p|") ** 2

    assert torch.equal(again, transformed)
    assert torch.allclose(replayed, transformed, atol=1e-6)
    # A padded slot that acquired an energy would be read as a constituent downstream.
    assert torch.equal(transformed[:, 11:], batch[:, 11:])
    # Within the float32 rounding of the matrices, relative to E^2.
    assert ((mass - MASS**2).abs() <= 1e-5 * energy**2).all()
    for name in unchanged:
        before, after = _physical(batch, name), _physical(transformed, name)
        assert torch.allclose(before, after, atol=1e-6, rtol=1e-6), name
    for name in moved:
        assert (
            _physical(transformed, name) - _physical(batch, name)
        ).abs().max() > 1e-3
    if group == "lorentz":
        # Only for a massless constituent is a beam boost a rigid shift in eta, by minus
        # the rapidity drawn for its jet, uniform within the bound.
        massless = record_jets()
        torch.manual_seed(0)
        boosted, xi = make_drawing_transform(group, RECORD_COLUMNS, None, 2.0)(massless)
        d_eta = boosted[:, :11, 7] - massless[:, :11, 7]
        assert torch.allclose(d_eta, -xi[:, None].double().expand_as(d_eta), atol=1e-5)
        assert xi.abs().max() <= 2.0 and (xi - xi[0]).abs().max() > 1e-3


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: HLS4MLDataModule(method="equivariance"), "method"),
        (
            lambda: HLS4MLDataModule(augmentation_scheme="permute"),
            "augmentation_scheme",
        ),
        # 'none' is a data.augmentation_scheme but no transform and no SCL group.
        (lambda: make_transform("none", RECORD_COLUMNS), "lorentz_iso"),
        (lambda: SymmetryConsistencyLoss("none"), "symm_loss"),
        # Before anything is read: the shards do not exist.
        *(
            (
                lambda m=method, s=scheme: JetLevelDataModule(
                    shards_dir="/absent", method=m, augmentation_scheme=s
                ),
                "only the baseline",
            )
            for method, scheme in (
                ("symm_loss", "none"),
                ("augmentation", "so2"),
                ("symm_loss", "lorentz"),
            )
        ),
    ],
)
def test_an_unsupported_regime_is_refused(build, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize("exclude", [[], ["j_eta"]])
def test_the_jet_level_datamodule_standardises_on_the_training_split(
    jet_level_shards: Path, exclude: list[str]
) -> None:
    # Read back independently of the datamodule, in float64 as it standardises: every
    # split and fold with the record's own training statistics, j_ptfrac, which has no
    # spread, centred and left undivided.
    columns = [name for name in JET_LEVEL_COLUMNS if name not in exclude]
    raw = {}
    for split in JET_LEVEL_SIZES:
        paths = sorted(jet_level_shards.glob(f"{split}-*.parquet"))
        table = pa.concat_tables([pq.read_table(path) for path in paths])
        x = np.stack([table[name].to_numpy() for name in columns], axis=1)
        raw[split] = x.astype(np.float64), table["label"].to_numpy().astype(np.int64)
    mean, std = raw["train"][0].mean(axis=0), raw["train"][0].std(axis=0)
    std[0] = 1.0
    shards = str(jet_level_shards)
    record = JetLevelDataModule(shards_dir=shards, exclude=exclude, batch_size=16)
    folded = JetLevelDataModule(
        shards_dir=shards, exclude=exclude, n_folds=5, kfold_seed=7, fold_idx=3
    )
    record.setup()
    folded.setup()
    pool = tuple(
        np.concatenate(arrays) for arrays in zip(raw["train"], raw["validation"])
    )
    train_idx, _ = fold_indices(len(pool[1]), 5, 7, 3)
    served = {
        record.data_train: raw["train"],
        record.data_test: raw["test"],
        folded.data_train: tuple(array[train_idx] for array in pool),
        folded.data_test: raw["test"],
    }

    assert record.features == columns
    for dataset, (x, classes) in served.items():
        standardised = torch.from_numpy(((x - mean) / std).astype(np.float32))
        assert torch.equal(dataset.X, standardised.unsqueeze(1))
        assert torch.equal(dataset.y, torch.eye(5)[torch.from_numpy(classes)])
    assert next(iter(record.train_dataloader()))[0].shape == (16, 1, len(columns))


def test_composing_the_record_leaves_the_running_hydra_alone(tmp_path: Path) -> None:
    stand_in = tmp_path / "record"
    (stand_in / "configs").mkdir(parents=True)
    (stand_in / "configs" / "config.yaml").write_text("data:\n  nconstituents: 32\n")

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        outer = GlobalHydra.instance().hydra
        composed = compose_record_config(str(stand_in), ["data.nconstituents=150"])

        assert composed.data.nconstituents == 150
        assert GlobalHydra.instance().hydra is outer


@pytest.mark.slow
@needs_record
def test_the_shards_hold_the_loaders_jets_in_its_order() -> None:
    # The folds select the networks' jets for the jet-level rows only if the shards hold
    # them in the order of the loader's cache, which the networks read.
    revision = OmegaConf.load(ROOT / "configs" / "data" / "jet_level.yaml").revision
    networks = HLS4MLDataModule(
        revision=revision, cache_dir=CACHE_DIR, nconstituents=16
    )
    networks.prepare_data()
    jet_level = JetLevelDataModule(revision=revision)
    jet_level.setup()
    served = {
        "train": jet_level.data_train,
        "validation": jet_level.data_val,
        "test": jet_level.data_test,
    }

    assert len(jet_level.feature_names) == 53 and "j_eta" in jet_level.feature_names
    assert [dataset.n for dataset in served.values()] == [496_000, 124_000, 260_000]
    for split, dataset in served.items():
        cached = networks.record.load(split)
        # A padded slot is an all-zero row, which the loader's normalisation keeps at 0.
        counts = (cached.x != 0).any(dim=-1).sum(dim=1).numpy()
        lengths = [
            pc.list_value_length(pq.read_table(path, columns=["j1_pt"])["j1_pt"])
            for path in shard_paths(jet_level.shards_directory, split)
        ]
        stored = np.concatenate([length.to_numpy() for length in lengths])
        assert torch.equal(dataset.y.argmax(dim=1), cached.y.argmax(dim=1)), split
        assert np.array_equal(counts, np.minimum(stored, 16)), split
