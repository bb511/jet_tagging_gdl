import csv
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial
from torchmetrics.classification import MulticlassAccuracy

from src.callbacks.invariance import clean_metrics
from src.symmetry import TABULATED_FAMILIES, TABULATED_GROUPS, key
from src.utils.results_table import ResultsTable
from tests.conftest import (
    BASELINE_ONLY,
    CPU_ROWS,
    EMBEDDED,
    FOUR_MOMENTUM,
    JET_LEVEL,
    MODELS,
    QUANTISED,
    ROOT,
    load_script,
)

generator = load_script("scripts/optuna/make_train_catalogues.py")
LIB = ROOT / "scripts" / "cluster" / "lib.sh"
STUDY = "mlp_nconst16_augmentation_none"
# Each completed trial carries its own value as its one parameter, so that a test sees
# whose parameters came back.
VALUES = CategoricalDistribution((0.5, 0.8, 0.9))


def _blocks(catalogue: Path) -> list[dict[str, str]]:
    # lib.sh's own parser, not a second implementation of it here, so that the two
    # cannot disagree about what a block is; each block becomes its overrides.
    script = f'. "{LIB}"; extract_cmds "$1"'
    found = subprocess.run(
        ["bash", "-c", script, "_", str(catalogue)],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in found.stdout.splitlines() if line.strip()]
    return [
        dict(t.split("=", 1) for t in shlex.split(line) if "=" in t) for line in lines
    ]


def _variant(block: dict[str, str]) -> str:
    method, scheme = block["data.method"], block["data.augmentation_scheme"]
    if (method, scheme) == ("augmentation", "none"):
        return "baseline"
    return f"{'aug' if method == 'augmentation' else 'symm'}_{scheme}"


def test_the_generator_names_the_rows_the_tests_do() -> None:
    # The generator keeps lists of its own, since it runs without the test suite; a row
    # added to one side and not the other would otherwise go unnoticed.
    assert tuple(generator.MODELS) == MODELS
    assert generator.EMBEDDED == set(EMBEDDED)
    assert generator.BASELINE_ONLY == set(BASELINE_ONLY)
    assert tuple(generator.JET_LEVEL) == JET_LEVEL
    assert generator.CPU_ROWS == set(CPU_ROWS)
    assert tuple(generator.QUANTISED) == QUANTISED


@pytest.mark.parametrize("features", [None, ("pt", "eta", "phi")])
def test_every_generated_block_parses_back_through_lib_sh(
    features: tuple[str, ...] | None, tmp_path: Path
) -> None:
    # No study database: every block is transferred from its experiment config, and
    # the render depends on nothing outside the repository. A feature set has no
    # catalogue of the embedded, jet-level or quantised rows.
    dbs, out = tmp_path / "optuna", tmp_path / "catalogues"
    dbs.mkdir()
    generator._generate(dbs, out, ROOT, generator.Layout(features=features), False)
    label = f"{'_'.join(features)}/" if features else ""
    rows = set(MODELS) - set(EMBEDDED) if features else {*MODELS, *JET_LEVEL}

    assert {p.name for p in out.iterdir()} == {
        f"run{r}_{t}.sh" for r in rows for t in ("train", "eval")
    }
    for row in rows:
        train, evaluate = (
            _blocks(out / f"run{row}_{task}.sh") for task in ("train", "eval")
        )
        grid = generator.NCONSTITUENTS_OF[row]
        assert (
            len(train) == len(evaluate) == len(grid) * len(generator.VARIANTS_OF[row])
        )
        for trained, evaluated in zip(train, evaluate):
            n = int(trained["experiment"].removeprefix(f"hls4ml/{row}_"))
            cell = f"{label}{row}/nconst{n}/{_variant(trained)}"
            assert n in grid
            assert trained["hydra.run.dir"] == f"logs/train/hls4ml/{cell}"
            assert evaluated["hydra.run.dir"] == f"logs/eval/hls4ml/{cell}"
            assert evaluated["checkpoint_manifest_path"] == trained["hydra.run.dir"]
            assert evaluated["results_table.invariance_metric"] == "tv"
            for block in (trained, evaluated):
                assert block.get("data.features") == (
                    "[pt,eta,phi]" if features else None
                )
                # The submitters' --gpu and --gpus rewrite trainer.devices alone.
                device = ("cpu", None) if row in CPU_ROWS else ("gpu", "[0]")
                assert (block["trainer"], block.get("trainer.devices")) == device


def _study(db: Path, rows: list[float | TrialState]) -> Path:
    # A float is a completed trial with that value; a RUNNING row is left open through
    # ask(), as a killed driver leaves one.
    study = optuna.create_study(
        study_name=STUDY, storage=f"sqlite:///{db}", direction="maximize"
    )
    for row in rows:
        if row == TrialState.RUNNING:
            study.ask()
        elif row == TrialState.FAIL:
            study.add_trial(create_trial(state=TrialState.FAIL))
        else:
            trial = create_trial(
                params={"x": row}, distributions={"x": VALUES}, value=row
            )
            study.add_trial(trial)
    return db


def test_best_trial_keeps_to_the_first_200_completed_trials(tmp_path: Path) -> None:
    # The overall best, 0.9, is trial 230, past the budget; within it 0.8 is shared by
    # trials 120 and 150, and the earlier one wins.
    peaks = {120: 0.8, 150: 0.8, 230: 0.9}
    tied = _study(tmp_path / "tied.db", [peaks.get(n, 0.5) for n in range(250)])
    # A FAIL and an open RUNNING row before every 50th completed trial put the 200th
    # completed trial at number 207 and the 201st at 210; counting every row would end
    # the window at number 199, where no trial beats 0.5.
    rows = []
    for k in range(201):
        rows += [TrialState.FAIL, TrialState.RUNNING] if k % 50 == 0 else []
        rows.append({199: 0.8, 200: 0.9}.get(k, 0.5))
    padded = _study(tmp_path / "padded.db", rows)
    empty = _study(tmp_path / "empty.db", [TrialState.FAIL, TrialState.RUNNING])

    assert generator.best_trial(tied, STUDY) == (120, 0.8, {"x": "0.8"})
    assert generator.best_trial(padded, STUDY) == (207, 0.8, {"x": "0.8"})
    assert generator.best_trial(empty, STUDY) is None
    assert generator.best_trial(empty, "absent") is None


def _submit(script: str, *args: str) -> subprocess.CompletedProcess:
    # From the repository root, which the submitters require; --dry-run needs neither
    # the env nor the data.
    command = ["bash", f"scripts/cluster/{script}", "--dry-run", *args]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize(
    "script, catalogue",
    [
        ("submit_train.sh", "runmlp_train.sh"),
        ("submit_eval.sh", "runmlp_eval.sh"),
        ("submit_search.sh", "runmlp_search.sh"),
    ],
)
def test_the_submitters_select_blocks_by_index_and_label(
    script: str, catalogue: str
) -> None:
    # An all-digit selector is an index alone: '1' must not match nconst16 as well.
    selection = ("--only", "1,nconst150_symm_so2", "--gpu", "2")
    found = _submit(script, *selection, f"scripts/hls4ml/{catalogue}", "seed=7")

    commands = [line for line in found.stdout.splitlines() if line.startswith("[")]
    assert found.returncode == 0, found.stderr
    assert [line.split("]")[0] for line in commands] == ["[1/21", "[19/21"]
    assert all(
        "trainer.devices=[2]" in line and line.endswith(" seed=7") for line in commands
    )


def test_submit_search_refuses_a_feature_block_without_its_own_storage(
    tmp_path: Path,
) -> None:
    # The study name ignores the feature set, so its trials would join the
    # four-momentum study of the same name.
    text = (ROOT / "scripts/hls4ml/pt_eta_phi/runmlp_search.sh").read_text()
    catalogue = tmp_path / "runmlp_search.sh"
    lines = [line for line in text.splitlines() if "hydra.sweeper.storage" not in line]
    catalogue.write_text("\n".join(lines))

    found = _submit("submit_search.sh", "--only", "1", str(catalogue))

    assert found.returncode == 2 and "no hydra.sweeper.storage" in found.stderr
    assert not any(line.startswith("[") for line in found.stdout.splitlines())


def test_submit_train_lays_the_folds_out_under_the_main_run() -> None:
    folds = ("--only", "1", "--folds", "2", "--gpus", "0,MIG-x")
    found = _submit("submit_train.sh", *folds, "scripts/hls4ml/runmlp_train.sh")

    processes = [
        line.split(": ", 1)
        for line in found.stdout.splitlines()
        if line.startswith("[1/21]")
    ]
    base = "logs/train/hls4ml/mlp/nconst16/baseline"
    assert [label for label, _ in processes] == [
        "[1/21] main",
        "[1/21] fold_1",
        "[1/21] fold_2",
    ]
    (_, main), (_, fold_1), (_, fold_2) = processes
    assert main.endswith(" data.n_folds=1") and "trainer.devices=[0]" in main
    # A UUID of --gpus goes into CUDA_VISIBLE_DEVICES, an index into the trainer; the
    # runner expands $parent, the main run's MLflow run, off its manifest.
    assert (
        fold_1.startswith("CUDA_VISIBLE_DEVICES=MIG-x ")
        and "trainer.devices=[0]" in fold_1
    )
    for j, fold in ((1, fold_1), (2, fold_2)):
        layout = f"fold={j} data.n_folds=2 hydra.run.dir={base}/fold_{j}"
        assert fold.endswith(f"{layout} parent_run_id=$parent")
    assert not fold_2.startswith("CUDA_VISIBLE_DEVICES")


def _results_row(directory: Path, scheme: str, tv: float, acc: float) -> Path:
    # One row in the frozen layout, written by the results table itself: three folds
    # that differ by a jitter common to both rows, every invariance family at tv.
    folds = []
    for fold in range(3):
        jitter = 0.001 * (fold - 1)
        clean = {"rejection_es30": 40.0, "tpr_eb1": 0.4, "tpr_eb10": 0.5, "auc": 0.9}
        values = {key(None, f): v + jitter for f, v in clean.items()}
        for family in ("tv", *TABULATED_FAMILIES):
            values |= {key(g, family): tv + jitter for g in TABULATED_GROUPS}
        costs = {"flops_total": 1000.0, "ebops_total": 64000.0}
        folds.append(values | costs | {"test/acc": acc + jitter})
    path = str(directory / "results_table.csv")
    table = ResultsTable(
        path, "augmentation", "mlp", scheme, 16, features=FOUR_MOMENTUM
    )
    table.write(folds)
    return table.output_path


def _main(monkeypatch: pytest.MonkeyPatch, script: str, *args: object) -> None:
    module = load_script(f"scripts/analysis/{script}.py")
    monkeypatch.setattr(sys, "argv", [f"{script}.py", *map(str, args)])
    module.main()


def test_the_analysis_scripts_read_a_two_row_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, figures = tmp_path / "results", tmp_path / "figures"
    baseline = _results_row(results, "none", tv=0.2, acc=0.70)
    run = _results_row(results, "so2", tv=0.1, acc=0.75)
    deltas = run.parent / "deltas_augmentation_vs_baseline.csv"

    _main(monkeypatch, "compare_to_baseline", run, baseline)
    _main(monkeypatch, "compare_to_baseline", run, baseline, "--baseline-model", "mlp")
    _main(monkeypatch, "make_latex_table", run, baseline, "--deltas", deltas)
    drawn = ("--features", "px_py_pz_e", "--deltas", deltas, "--out-dir", figures)
    _main(monkeypatch, "make_figures", "--results-dir", results, *drawn)
    _main(monkeypatch, "make_baseline_figure", baseline, "--out-dir", figures)

    rows = list(csv.DictReader(deltas.open()))
    delta = {(row["group"], row["metric"]): row for row in rows}
    named = run.parent / "deltas_augmentation_vs_baseline_mlp.csv"
    assert len(list(csv.DictReader(named.open()))) == len(rows)
    # Positive always means the run is the better model: higher accuracy, retention and
    # stable fraction, less TV and fewer flips. The diagnostics are never compared.
    assert float(delta[("", "test_acc")]["delta_mean"]) == pytest.approx(0.05)
    assert delta[("", "test_acc")]["delta_folds"] == "0.050000;0.050000;0.050000"
    assert float(delta[("so2", "tv")]["delta_mean"]) == pytest.approx(0.1)
    assert float(delta[("so2", "flip")]["delta_mean"]) == pytest.approx(0.1)
    assert float(delta[("so2", "flip_stable")]["delta_mean"]) == pytest.approx(-0.1)
    assert not any(metric.startswith(("efficiency_", "fpr_")) for _, metric in delta)
    for written in (run.with_suffix(".tex"), baseline.with_suffix(".tex")):
        assert written.stat().st_size > 0
    for name in ("fig_augmentation.svg", "fig_baseline.svg"):
        assert (figures / name).stat().st_size > 0


def test_compare_outputs_passes_an_identical_candidate_alone(tmp_path: Path) -> None:
    compare_outputs = load_script("scripts/analysis/compare_outputs.py")
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 5, 5000).astype(np.int8)
    logits = (rng.normal(size=(5000, 5)) + 2.0 * np.eye(5)[truth]).astype(np.float32)
    noisy = (logits + rng.normal(size=logits.shape)).astype(np.float32)
    table = tmp_path / "fidelity.csv"
    for label, values in (("reference", logits), ("same", logits), ("noisy", noisy)):
        np.savez(tmp_path / label, logits_clean=values, labels=truth)
    for label in ("same", "noisy"):
        files = [str(tmp_path / f"{name}.npz") for name in ("reference", label)]
        argv = ["--reference", files[0], "--candidate", files[1], "--label", label]
        compare_outputs.main([*argv, "--csv", str(table)])
    rows = list(csv.DictReader(table.open()))
    # A class never labelled and a class never predicted: the macro accuracy of
    # torchmetrics, the results table's test_acc, averages over the classes seen.
    seen, never = truth.copy(), logits.copy()
    seen[seen == 3] = 0
    never[:, 4] -= 100.0
    expected = MulticlassAccuracy(num_classes=5)(
        torch.from_numpy(never), torch.from_numpy(seen.astype(np.int64))
    )
    figures = clean_metrics(logits, truth)

    assert [(row["label"], row["verdict"]) for row in rows] == [
        ("same", "PASS"),
        ("noisy", "FAIL"),
    ]
    assert all(float(v) == 0.0 for name, v in rows[0].items() if name.endswith("_rel"))
    assert float(rows[0]["roc_gap_max"]) == 0.0 < float(rows[1]["roc_gap_max"])
    assert float(rows[0]["auc_candidate"]) == pytest.approx(figures[key(None, "auc")])
    assert compare_outputs.accuracies(never, seen)["performance/acc"] == pytest.approx(
        expected.item(), abs=1e-6
    )
