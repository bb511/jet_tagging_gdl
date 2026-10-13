from pathlib import Path

import pytest

from src.utils.checkpoints import Manifest, ManifestRun, merge_manifests, write_manifest
from tests.conftest import load_script

picks = load_script("scripts/experiments/quantised_picks.py")

# epoch: (logged EBOPs, logged validation accuracy, traced EBOPs). Epoch 13 is below the
# front's 0.5, 14 is dominated by 11, and 12 ties 16's EBOPs with the higher accuracy.
QUANTISED_EPOCHS = {
    10: (300_000, 0.74, 301_000),
    11: (200_000, 0.73, 203_000),
    12: (210_000, 0.735, 211_000),
    13: (150_000, 0.40, 151_000),
    14: (250_000, 0.72, 252_000),
    15: (100_000, 0.70, 104_000),
    16: (210_000, 0.731, 212_000),
}
LOG_PREFIX = "[2026-10-13 00:00:00,000][src.algorithms.keras_base][INFO] - [rank: 0] "


def _quantised_run(root: Path, fold: int | None = None) -> Path:
    # A training as train.py leaves it, each epoch's checkpoint, both of its INFO lines
    # and a sanity-check validation line before epoch 15's own, and the evaluation of
    # every epoch under <root>/evaluations.
    run = root if fold is None else root / f"fold_{fold}"
    (run / "checkpoints").mkdir(parents=True)
    lines = [f"{LOG_PREFIX}[epoch  15]   val | loss nan acc 0.2000 best 0.2000"]
    for epoch, (ebops, acc, traced) in QUANTISED_EPOCHS.items():
        step = f"[epoch {epoch:>4d}] EBOPs {ebops}, beta 1e-07, lr 0.003"
        lines.append(LOG_PREFIX + step)
        lines.append(f"{LOG_PREFIX}[epoch {epoch:>3d}]   val | loss 0.8 acc {acc:.4f}")
        (run / "checkpoints" / f"epoch_{epoch:03d}.ckpt").touch()
        export = run / "evaluations" / f"epoch_{epoch:03d}" / "outputs" / "main_hgq"
        export.mkdir(parents=True)
        (export / "export.json").write_text(f'{{"ebops": {traced}}}')
    (run / "train.log").write_text("\n".join(lines) + "\n")
    label = "main" if fold is None else f"fold_{fold}"
    entry = ManifestRun(label, fold, str(run / "checkpoints" / "epoch_016.ckpt"), 0.7)
    write_manifest(Manifest(0 if fold is None else 1, str(run), "x", [entry]), run)
    return run


def test_the_quantised_front_is_their_pareto_front_of_validation_and_ebops(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run = _quantised_run(tmp_path)
    picks.main(["front", str(run)])

    assert capsys.readouterr().out.split() == ["15", "11", "12", "10"]


def test_the_quantised_picks_write_manifests_eval_merges(tmp_path: Path) -> None:
    main, fold = _quantised_run(tmp_path), _quantised_run(tmp_path, fold=1)
    out = tmp_path / "picks" / "target"
    for run in (main, fold):
        picks.main(["pick", str(run), str(run / "evaluations"), "206000", str(out)])
    merged = merge_manifests(out)
    nearest = [Path(run.checkpoint_path).name for run in merged.runs]

    # Traced, 203 000 lies nearer 206 000 than 211 000; the logged EBOPs would pick 12.
    assert nearest == ["epoch_011.ckpt", "epoch_011.ckpt"]
    assert [run.fold_label for run in merged.runs] == ["main", "fold_1"]
    assert merged.n_folds == 1 and merged.experiment_name == "x"
    empty = _quantised_run(tmp_path / "empty")
    (empty / "train.log").write_text("")
    with pytest.raises(SystemExit, match="no epoch on the front"):
        picks.pick(empty, empty / "evaluations", 1.0)
