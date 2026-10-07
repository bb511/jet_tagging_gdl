import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_curve
from torchmetrics.classification import MulticlassAccuracy

from src.callbacks.invariance import clean_metrics
from tests.conftest import ROOT

# scripts/ is not a package, so the module is loaded by path, with its own directory put
# on the import path first: running it as a script puts it there, and it imports common
# from beside itself.
ANALYSIS = ROOT / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

_SPEC = importlib.util.spec_from_file_location(
    "compare_outputs", ANALYSIS / "compare_outputs.py"
)
compare_outputs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_outputs)

N_JETS = 5000
CLASSES = compare_outputs.CLASS_NAMES


def _outputs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # Gaussian logits with a head start of 2 for each jet's own class, in the probe's
    # dtypes.
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, N_JETS).astype(np.int8)
    logits = rng.normal(size=(N_JETS, 5)) + 2.0 * np.eye(5)[labels]
    return logits.astype(np.float32), labels


def _files(tmp_path: Path, candidate: np.ndarray) -> list[str]:
    logits, labels = _outputs()
    reference, other = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    np.savez(reference, logits_clean=logits, labels=labels)
    np.savez(other, logits_clean=candidate.astype(np.float32), labels=labels)
    return ["--reference", str(reference), "--candidate", str(other)]


def _row(tmp_path: Path, candidate: np.ndarray, *extra: str) -> dict:
    args = compare_outputs.parse_args([*_files(tmp_path, candidate), *extra])
    return compare_outputs.compare(args)[0]


def _noisy(sigma: float, seed: int = 2) -> np.ndarray:
    logits, _ = _outputs()
    return logits + sigma * np.random.default_rng(seed).normal(size=logits.shape)


def test_an_identical_candidate_passes_with_no_difference(tmp_path: Path) -> None:
    logits, labels = _outputs()
    row = _row(tmp_path, logits.copy())

    relative = [value for name, value in row.items() if name.endswith("_rel")]
    passes = [value for name, value in row.items() if name.endswith("_pass")]
    assert len(relative) == 5 * 6 and all(value == 0.0 for value in relative)
    assert len(passes) == 3 * 6 and all(passes)
    assert row["verdict"] == "PASS"
    assert row["roc_gap_max"] == 0.0
    expected = clean_metrics(logits, labels)
    for family in compare_outputs.PERFORMANCE_FAMILIES:
        for scope in compare_outputs.SCOPES:
            name = compare_outputs.column(family, scope, "candidate")
            assert row[name] == expected[compare_outputs.key(None, family, scope)]


@pytest.mark.parametrize("shuffled", range(5))
def test_shuffling_the_logits_of_one_class_fails_its_working_points(
    tmp_path: Path, shuffled: int
) -> None:
    logits, _ = _outputs()
    rng = np.random.default_rng(1)
    logits[:, shuffled] = rng.permutation(logits[:, shuffled])
    row = _row(tmp_path, logits)

    name = CLASSES[shuffled]
    for family in compare_outputs.JUDGED:
        assert not row[compare_outputs.column(family, name, "pass")]
    assert row["verdict"] == "FAIL"
    assert row[f"roc_gap_{name}"] == row["roc_gap_max"] > 0.1


def test_noise_within_the_tolerance_passes_and_beyond_it_fails(tmp_path: Path) -> None:
    small = _row(tmp_path, _noisy(1e-4))
    large = _row(tmp_path, _noisy(1.0))

    assert small["verdict"] == "PASS"
    assert 0.0 < small["roc_gap_max"] < 0.01
    assert large["verdict"] == "FAIL"
    assert compare_outputs.failed(large) == list(compare_outputs.JUDGED)


@pytest.mark.parametrize(
    "unlabelled, unpredicted", [((), ()), ((3,), ()), ((2, 4), (4,))]
)
def test_the_macro_accuracy_is_that_of_torchmetrics(
    unlabelled: tuple[int, ...], unpredicted: tuple[int, ...]
) -> None:
    logits, labels = _outputs()
    labels[np.isin(labels, unlabelled)] = 0
    logits[:, list(unpredicted)] -= 100.0
    tensors = torch.from_numpy(logits), torch.from_numpy(labels.astype(np.int64))

    accuracies = compare_outputs.accuracies(logits, labels)

    expected = MulticlassAccuracy(num_classes=5)(*tensors).item()
    assert accuracies["performance/acc"] == pytest.approx(expected, abs=1e-6)
    if not unlabelled:
        per_class = MulticlassAccuracy(num_classes=5, average=None)(*tensors)
        observed = [accuracies[f"performance/acc_{name}"] for name in CLASSES]
        assert observed == pytest.approx(per_class.tolist(), abs=1e-6)


def test_the_roc_gap_is_the_largest_difference_of_the_step_functions() -> None:
    # Signal, signal, background, background: perfect in the reference; the candidate
    # ranks a background jet second, so its TPR is 1/2 where the reference's is 1, at
    # every FPR below 1/2. That one background jet opens a step 1/n_B = 1/2 wide: a
    # floor of 1/4 still reads it, one of 1/2 leaves it out.
    signal = np.array([True, True, False, False])
    reference = compare_outputs.roc(signal, np.array([4.0, 3.0, 2.0, 1.0]))
    candidate = compare_outputs.roc(signal, np.array([4.0, 2.0, 3.0, 1.0]))

    assert compare_outputs.roc_gap(reference, reference) == 0.0
    assert compare_outputs.roc_gap(reference, candidate) == 0.5
    assert compare_outputs.roc_gap(reference, candidate, 0.0) == 0.5
    assert compare_outputs.roc_gap(reference, candidate, 0.25) == 0.5
    assert compare_outputs.roc_gap(reference, candidate, 0.5) == 0.0
    assert compare_outputs.tpr_at(candidate, np.array([0.0, 0.49, 0.5])).tolist() == [
        0.5,
        0.5,
        1.0,
    ]


def test_the_roc_gap_is_zero_for_equal_scores_and_positive_otherwise() -> None:
    logits, labels = _outputs()
    reference = compare_outputs.roc_curves(logits, labels)
    same = compare_outputs.roc_curves(logits.copy(), labels)
    perturbed = compare_outputs.roc_curves(_noisy(0.3), labels)

    assert all(compare_outputs.roc_gap(r, s) == 0.0 for r, s in zip(reference, same))
    assert all(
        compare_outputs.roc_gap(r, p) > 0.0 for r, p in zip(reference, perturbed)
    )


def test_the_curves_hold_every_threshold() -> None:
    logits, labels = _outputs()
    curves = compare_outputs.roc_curves(logits, labels)
    scores = compare_outputs.one_vs_rest(logits)

    for c, (fpr, tpr) in enumerate(curves):
        expected = roc_curve(labels == c, scores[:, c], drop_intermediate=False)
        assert np.array_equal(fpr, expected[0]) and np.array_equal(tpr, expected[1])
        assert len(fpr) == len(np.unique(scores[:, c])) + 1


def test_the_csv_gets_one_row_per_comparison_under_its_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _files(tmp_path, _noisy(1e-3))
    table = tmp_path / "table" / "fidelity.csv"

    for label, floor in (("beta_1e-7", []), ("beta_1e-6", ["--fpr-floor", "0"])):
        argv = [*files, "--label", label, "--csv", str(table), *floor]
        assert compare_outputs.main(argv) == 0

    rows = list(csv.DictReader(table.open()))
    assert [row["label"] for row in rows] == ["beta_1e-7", "beta_1e-6"]
    head = ["label", "reference", "candidate", "n_jets", "tolerance", "fpr_floor"]
    assert list(rows[0])[:7] == [*head, "verdict"]
    columns = {"acc_rel", "tpr_eb1_pass", "auc_t_candidate", "roc_gap_max"}
    assert columns <= set(rows[0])
    assert rows[0]["n_jets"] == str(N_JETS)
    assert [row["fpr_floor"] for row in rows] == ["0.001", "0.0"]
    assert float(rows[0]["roc_gap_max"]) <= float(rows[1]["roc_gap_max"])
    assert "at FPR >= 0.001" in capsys.readouterr().out


@pytest.mark.parametrize("floor", ["-0.1", "1"])
def test_a_floor_outside_the_fpr_range_is_refused(tmp_path: Path, floor: str) -> None:
    argv = [*_files(tmp_path, _noisy(1e-3)), "--fpr-floor", floor]

    with pytest.raises(SystemExit):
        compare_outputs.parse_args(argv)


def test_a_csv_of_other_columns_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "other.csv"
    table.write_text("model,test_acc\nvn,0.72\n")
    argv = [*_files(tmp_path, _noisy(1e-3)), "--csv", str(table)]

    with pytest.raises(SystemExit):
        compare_outputs.main(argv)
    assert table.read_text() == "model,test_acc\nvn,0.72\n"


def test_other_labels_are_refused(tmp_path: Path) -> None:
    argv = _files(tmp_path, _outputs()[0])
    other = tmp_path / "shuffled.npz"
    logits, labels = _outputs()
    np.savez(other, logits_clean=logits, labels=np.roll(labels, 1))

    with pytest.raises(SystemExit):
        compare_outputs.main([*argv[:2], "--candidate", str(other)])


def test_the_figure_is_written(tmp_path: Path) -> None:
    figure = tmp_path / "figures" / "roc.pdf"

    compare_outputs.main([*_files(tmp_path, _noisy(0.1)), "--figure", str(figure)])

    assert figure.is_file() and figure.stat().st_size > 0
