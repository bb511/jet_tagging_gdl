import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torchmetrics.classification import MulticlassAccuracy

from src.callbacks.invariance import clean_metrics
from src.data.hls4ml_datamodule import HLS4MLDataModule, fold_indices
from tests.conftest import CACHE_DIR, ROOT, needs_record, results_header

# scripts/ is not a package, so the script is loaded by path.
_SPEC = importlib.util.spec_from_file_location(
    "ceiling", ROOT / "scripts" / "jet_level" / "ceiling.py"
)
ceiling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ceiling)

CPU = torch.device("cpu")
# The columns the jet-level rows report: the test accuracy and the clean working points.
GROUPLESS_FOLDS = {
    "test_acc_folds",
    "rejection_es30_folds",
    "tpr_eb1_folds",
    "tpr_eb10_folds",
    "auc_folds",
}


def _run(tmp_path: Path, epochs: int = 1) -> ceiling.Run:
    return ceiling.Run(
        device=CPU,
        epochs=epochs,
        quick=False,
        select=False,
        kfold_seed=7,
        run_dir=tmp_path,
        results_dir=tmp_path / "results",
    )


def _jets(n_jets: int, n_observables: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Every class present, so that the macro accuracy has five terms.
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_jets, n_observables)).astype(np.float32)
    return x, np.arange(n_jets, dtype=np.int64) % 5


def test_the_two_rows_drop_the_jet_rapidity_alone() -> None:
    names = [f"j_observable_{i}" for i in range(52)]
    names.insert(2, "j_eta")

    assert set(ceiling.ROWS) == {"jet_level_all53", "jet_level_noeta52"}
    assert ceiling.columns_of(names, "jet_level_all53") == list(range(53))
    assert ceiling.columns_of(names, "jet_level_noeta52") == [0, 1, *range(3, 53)]


def test_the_accuracy_is_the_macro_mean_over_the_classes() -> None:
    # 80 gluon jets, all called gluons, and five of each other class, one of them right.
    labels = torch.tensor([0] * 80 + [c for c in range(1, 5) for _ in range(5)])
    predicted = labels.clone()
    predicted[80:] = torch.tensor(
        [c if i == 0 else 0 for c in range(1, 5) for i in range(5)]
    )
    logits = 3.0 * torch.eye(5)[predicted]
    stateful = MulticlassAccuracy(num_classes=5)
    for chunk in (slice(0, 37), slice(37, 100)):
        stateful.update(logits[chunk], labels[chunk])

    macro = ceiling.macro_accuracy(logits, labels)

    # (1 + 4 * 0.2) / 5, where the micro mean would be (80 + 4) / 100.
    assert macro == pytest.approx(0.36)
    assert macro == pytest.approx(float(stateful.compute()))
    assert float((predicted == labels).float().mean()) == pytest.approx(0.84)


def test_the_standardisation_reads_the_training_jets_alone_and_spares_constant_columns(
    tmp_path: Path, monkeypatch
) -> None:
    x, y = _jets(60, 3, seed=0)
    x[:, 2] = 1.0
    test = _jets(20, 3, seed=1)
    splits = {"pool": (x, y), "test": (test[0] + 100.0, test[1])}
    seen = []
    standardiser = ceiling.standardiser
    monkeypatch.setattr(
        ceiling,
        "standardiser",
        lambda x_train: seen.append(x_train) or standardiser(x_train),
    )

    metrics, summary = ceiling.run_fold(
        "jet_level_all53", splits, ceiling.Recipe((4,), 0.0, 1e-2), 2, _run(tmp_path)
    )

    train_idx, val_idx = fold_indices(60, 5, 7, 1)
    assert len(seen) == 1 and np.array_equal(seen[0], x[train_idx])
    assert summary["jets"] == {"train": 48, "validation": 12}
    mean, std = standardiser(x[train_idx])
    np.testing.assert_allclose(mean, x[train_idx].astype(np.float64).mean(axis=0))
    # The constant column is centred to zero rather than divided by a zero spread.
    assert std[2] == 1.0
    standardised, _ = ceiling.tensors(x[train_idx], y[train_idx], (mean, std), CPU)
    assert torch.all(standardised[:, 2] == 0.0)
    assert torch.isfinite(standardised).all()
    with np.load(tmp_path / "outputs" / "jet_level_all53" / "fold_2.npz") as stored:
        assert stored["logits_clean"].shape == (20, 5)
        assert stored["labels"].dtype == np.int8
    # The fold dict eval.py would collect: the accuracy and the 24 groupless keys.
    assert (
        metrics.keys() - {"test/acc"}
        == clean_metrics(np.zeros((20, 5)), test[1]).keys()
    )


def test_the_order_guard_refuses_swapped_labels_and_swapped_counts() -> None:
    labels = np.array([0, 1, 2, 3, 4, 0])
    counts = np.array([3, 5, 150, 150, 7, 2])
    ceiling.check_split("test", (labels, counts), (labels.copy(), counts.copy()))

    with pytest.raises(RuntimeError, match="labels of 2 jets, the first at index 0"):
        ceiling.check_split(
            "test", (labels[[1, 0, 2, 3, 4, 5]], counts), (labels, counts)
        )
    with pytest.raises(RuntimeError, match="constituent counts"):
        ceiling.check_split(
            "test", (labels, counts[[0, 2, 1, 3, 4, 5]]), (labels, counts)
        )
    with pytest.raises(RuntimeError, match="5 jets, the shards 6"):
        ceiling.check_split("test", (labels[:5], counts[:5]), (labels, counts))


def test_training_keeps_the_best_validation_weights(monkeypatch) -> None:
    x, y = _jets(200, 6, seed=0)
    scale = ceiling.standardiser(x[:150])
    training = ceiling.tensors(x[:150], y[:150], scale, CPU)
    validation = ceiling.tensors(x[150:], y[150:], scale, CPU)
    snapshots = []
    train_epoch = ceiling.train_epoch

    def recorded(model, *args) -> None:
        train_epoch(model, *args)
        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})

    # The second of three epochs scores best, so the weights kept are neither the first
    # epoch's nor the last's.
    scores = iter([0.3, 0.5, 0.4])
    monkeypatch.setattr(ceiling, "train_epoch", recorded)
    monkeypatch.setattr(ceiling, "macro_accuracy", lambda logits, labels: next(scores))

    model, best, epochs_run = ceiling.fit(
        ceiling.Recipe((8,), 0.0, 1e-2), training, validation, epochs=3
    )

    assert (best.epoch, best.acc, epochs_run) == (2, 0.5, 3)
    state = model.state_dict()
    assert all(torch.equal(state[name], snapshots[1][name]) for name in state)
    assert not all(torch.equal(state[name], snapshots[2][name]) for name in state)


def test_a_written_row_has_the_networks_header_and_no_invariance_or_cost(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    labels = np.arange(500) % 5
    folds = []
    for _ in range(5):
        logits = rng.normal(size=(500, 5)) + 2.0 * np.eye(5)[labels]
        accuracy = ceiling.macro_accuracy(
            torch.from_numpy(logits), torch.from_numpy(labels)
        )
        folds.append({"test/acc": accuracy} | clean_metrics(logits, labels))

    path = ceiling.write_row(tmp_path, "jet_level_noeta52", folds)

    assert path == tmp_path / "jet_level" / "results_table_baseline.csv"
    with open(path, newline="") as table:
        header = next(csv.reader(table))
        table.seek(0)
        (row,) = list(csv.DictReader(table))
    assert header == results_header()
    assert (row["method"], row["model"]) == ("baseline", "jet_level_noeta52")
    assert (row["nconstituents"], row["folds_used"]) == ("150", "5")
    assert row["selected_inv_key"] == "invariance/mean_tv"
    for column in header:
        if column.endswith("_folds") and column not in GROUPLESS_FOLDS:
            assert row[column] == ";".join(["nan"] * 5), column
    assert all(
        len(row[c].split(";")) == 5 and "nan" not in row[c] for c in GROUPLESS_FOLDS
    )
    assert math.isnan(float(row["flops_mean"])) and math.isnan(float(row["ebops_mean"]))


@pytest.mark.slow
@needs_record
def test_the_shards_hold_the_loaders_jets_in_its_order(tmp_path: Path) -> None:
    revision = ceiling.data_config().revision
    data_dir = ceiling.shard_dir(revision)
    data, names = ceiling.read_record(data_dir)
    # The guard reads a cache and never builds one; the record tests build this one too.
    HLS4MLDataModule(
        revision=revision, cache_dir=CACHE_DIR, nconstituents=16
    ).prepare_data()

    outcome = ceiling.check_order(data, data_dir, revision, Path(CACHE_DIR), 16)

    assert outcome.startswith("passed")
    assert len(names) == 53 and "j_eta" in names
    sizes = [len(data[split][1]) for split in ceiling.SPLITS]
    assert sizes == [496_000, 124_000, 260_000]
    with pytest.raises(RuntimeError, match="No complete loader cache"):
        ceiling.check_order(data, data_dir, revision, tmp_path, 16)
    assert not any(tmp_path.iterdir())
