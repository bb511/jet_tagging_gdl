import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.checkpoints import Manifest, ManifestRun, merge_manifests, write_manifest
from tests.conftest import load_script

picks = load_script("scripts/experiments/quantised_picks.py")

# epoch: (logged EBOPs, logged validation accuracy, logged training accuracy, traced
# EBOPs, calibrated validation accuracy). Epoch 13's logged validation wrapped and its
# training accuracy keeps it a candidate; 14 is on neither logged front; 17's checkpoint
# was deleted; 12 ties 11's calibrated accuracy at higher EBOPs; 18 is the cheapest point
# of the validation fronts but collapsing, its training accuracy under 0.95 x 0.80.
QUANTISED_EPOCHS = {
    10: (300_000, 0.74, 0.80, 301_000, 0.75),
    11: (200_000, 0.73, 0.79, 203_000, 0.74),
    12: (210_000, 0.735, 0.79, 211_000, 0.74),
    13: (150_000, 0.40, 0.78, 151_000, 0.72),
    14: (250_000, 0.72, 0.76, 252_000, 0.73),
    15: (100_000, 0.70, 0.77, 104_000, 0.70),
    17: (120_000, 0.65, 0.775, 121_000, 0.71),
    18: (90_000, 0.69, 0.50, 91_000, 0.69),
}
UNEVALUATED, DELETED = 14, 17
# The calibrated front's epochs: (estimated latency in ns, LUTs, DSPs). Epoch 10 is over
# 100 ns.
HARDWARE = {
    10: (110.0, 200_000, 0),
    11: (95.0, 140_000, 0),
    13: (80.0, 100_000, 1229),
    15: (70.0, 60_000, 0),
}
LOG_PREFIX = "[2026-10-13 00:00:00,000][src.algorithms.keras_base][INFO] - [rank: 0] "
N_JETS, N_CLASSES = 100, 5


def _evaluation(outputs: Path, traced: float, accuracy: float) -> None:
    # What src/eval.py leaves: the labels by class index, logits right on the first
    # round(accuracy * N_JETS) jets, and the export with the traced EBOPs.
    (outputs / "main_hgq").mkdir(parents=True)
    labels = np.arange(N_JETS) % N_CLASSES
    shown = np.where(np.arange(N_JETS) < round(accuracy * N_JETS), labels, labels + 1)
    logits = np.eye(N_CLASSES)[shown % N_CLASSES]
    np.savez(outputs / "main.npz", logits_clean=logits, labels=labels.astype(np.int8))
    (outputs / "main_hgq" / "export.json").write_text(f'{{"ebops": {traced}}}')


def _quantised_run(root: Path, fold: int | None = None) -> Path:
    # A training as train.py leaves it, with both INFO lines of every epoch and a
    # sanity-check validation line before epoch 15's own, and the evaluation of every
    # candidate under <run>/evaluations.
    run = root if fold is None else root / f"fold_{fold}"
    (run / "checkpoints").mkdir(parents=True)
    lines = [f"{LOG_PREFIX}[epoch  15]   val | loss nan acc 0.2000 best 0.2000"]
    for epoch, (ebops, val, acc, traced, calibrated) in QUANTISED_EPOCHS.items():
        lines.append(f"{LOG_PREFIX}[epoch {epoch:>3d}]   val | loss 0.8 acc {val:.4f}")
        step = (
            f"[epoch {epoch:>4d}] EBOPs {ebops}, beta 1e-07, lr 0.003, train acc {acc}"
        )
        lines.append(LOG_PREFIX + step)
        if epoch != DELETED:
            (run / "checkpoints" / f"epoch_{epoch:03d}.ckpt").touch()
        if epoch not in (DELETED, UNEVALUATED):
            outputs = run / "evaluations" / f"epoch_{epoch:03d}" / "outputs"
            _evaluation(outputs, traced, calibrated)
        if epoch in HARDWARE:
            latency, lut, dsp = HARDWARE[epoch]
            estimate = {"latency_ns": latency, "lut": lut, "dsp": dsp}
            (outputs.parent / "estimate.json").write_text(json.dumps(estimate))
    (run / "train.log").write_text("\n".join(lines) + "\n")
    label = "main" if fold is None else f"fold_{fold}"
    entry = ManifestRun(label, fold, str(run / "checkpoints" / "epoch_016.ckpt"), 0.7)
    write_manifest(Manifest(0 if fold is None else 1, str(run), "x", [entry]), run)
    return run


def test_the_candidates_lie_on_either_logged_front_with_a_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run = _quantised_run(tmp_path)
    picks.main(["candidates", str(run)])

    assert capsys.readouterr().out.split() == ["10", "11", "12", "13", "15"]


def test_the_quantised_picks_follow_the_calibrated_front(tmp_path: Path) -> None:
    main, fold = _quantised_run(tmp_path), _quantised_run(tmp_path, fold=1)
    out = tmp_path / "picks" / "target"
    for run in (main, fold):
        picks.main(["pick", str(run), str(run / "evaluations"), "160000", str(out)])
    merged = merge_manifests(out)
    nearest = [Path(run.checkpoint_path).name for run in merged.runs]

    # Epoch 13 at 151 000 is nearest; the logged validation would have dropped it and
    # picked 11.
    assert nearest == ["epoch_013.ckpt", "epoch_013.ckpt"]
    assert picks.pick(main, main / "evaluations", 210_000) == (11, 0.74, 203_000.0)
    # Collapsing, epoch 18 would be the front's cheapest point and nearest 80 000.
    assert picks.pick(main, main / "evaluations", 80_000) == (15, 0.70, 104_000.0)
    assert [run.fold_label for run in merged.runs] == ["main", "fold_1"]
    assert merged.n_folds == 1 and merged.experiment_name == "x"
    empty = _quantised_run(tmp_path / "empty")
    (empty / "train.log").write_text("")
    with pytest.raises(SystemExit, match="no epoch on the front"):
        picks.pick(empty, empty / "evaluations", 1.0)


def test_the_hardware_picks_keep_the_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run = _quantised_run(tmp_path / "run")
    evaluations, out = run / "evaluations", tmp_path / "picks"
    picks.main(["front", str(run), str(evaluations)])
    picks.main(["pick-hw", str(run), str(evaluations), "100", str(out)])

    assert capsys.readouterr().out.split()[:4] == ["15", "13", "11", "10"]
    # Epoch 10 is the most accurate but over budget.
    assert Path(merge_manifests(out).runs[0].checkpoint_path).name == "epoch_011.ckpt"
    with pytest.raises(SystemExit, match="within 50.0 ns"):
        picks.pick_hardware(run, evaluations, 50.0)
