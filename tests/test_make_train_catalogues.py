import importlib.util
import re
import sys
from pathlib import Path

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial

from tests.conftest import BASELINE_ONLY, EMBEDDED, MODELS, ROOT

# scripts/ is not a package, so the generator is loaded by path, as the analysis scripts'
# tests load theirs. It is registered before it is executed because dataclasses resolves
# the string annotations of ``Layout`` through sys.modules[__name__].
GENERATOR = ROOT / "scripts" / "optuna" / "make_train_catalogues.py"
_SPEC = importlib.util.spec_from_file_location("make_train_catalogues", GENERATOR)
make_train_catalogues = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = make_train_catalogues
_SPEC.loader.exec_module(make_train_catalogues)

Layout = make_train_catalogues.Layout
LIB = ROOT / "scripts" / "cluster" / "lib.sh"
FEATURES = ("pt", "eta", "phi")
LABEL = "pt_eta_phi"
OVERRIDE = "data.features='[pt,eta,phi]'"
GRID = [
    (nconst, variant[0])
    for nconst in make_train_catalogues.NCONSTITUENTS
    for variant in make_train_catalogues.VARIANTS
]
STUDY = "mlp_nconst16_augmentation_none"
# Each completed trial carries its own value as its one parameter, so that a test sees
# whose parameters came back.
VALUES = CategoricalDistribution((0.5, 0.8, 0.9))


def _render(task: str, dbs: Path, layout: Layout) -> str:
    # MLP throughout: it carries the whole 3 x 7 grid, as DeepSets and JetFormer do.
    return make_train_catalogues.render("mlp", task, dbs, layout)


def _manifest_paths(catalogue: str) -> set[str]:
    # Whole tokens: the run directory is a prefix of its manifest file's path, so a
    # substring test would accept either form.
    return set(re.findall(r"checkpoint_manifest_path=(\S+)", catalogue))


def _empty_dbs(tmp_path: Path) -> Path:
    # No study database, so every block is transferred from its experiment config and
    # the render depends on nothing outside the repository.
    dbs = tmp_path / "optuna"
    dbs.mkdir()
    return dbs


def _study(db: Path, rows: list[float | TrialState]) -> None:
    # A float is a completed trial with that value. A RUNNING row is left open through
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
            params, distributions = {"x": row}, {"x": VALUES}
            study.add_trial(
                create_trial(params=params, distributions=distributions, value=row)
            )


def test_a_feature_set_puts_its_override_and_its_own_run_dirs_in_every_block(
    tmp_path: Path,
) -> None:
    dbs = _empty_dbs(tmp_path)
    layout = Layout(features=FEATURES)

    train = _render("train", dbs, layout)
    evaluate = _render("eval", dbs, layout)

    assert train.count(OVERRIDE) == len(GRID)
    assert evaluate.count(OVERRIDE) == len(GRID)
    manifests = _manifest_paths(evaluate)
    for nconst, label in GRID:
        cell = f"mlp/nconst{nconst}/{label}"
        assert f"hydra.run.dir=logs/train/hls4ml/{LABEL}/{cell}" in train
        assert f"logs/train/hls4ml/{LABEL}/{cell}" in manifests
        assert f"hydra.run.dir=logs/eval/hls4ml/{LABEL}/{cell}" in evaluate
    # Through lib.sh's own parser: the submitters must still find the whole grid.
    assert make_train_catalogues.count_commands(LIB, train) == len(GRID)
    assert make_train_catalogues.count_commands(LIB, evaluate) == len(GRID)


def test_a_feature_set_has_no_catalogue_for_the_embedded_architectures(
    tmp_path: Path,
) -> None:
    out = tmp_path / "catalogues"
    layout = Layout(features=FEATURES)

    make_train_catalogues._generate(_empty_dbs(tmp_path), out, ROOT, layout, False)

    assert sorted(path.name for path in out.iterdir()) == sorted(
        f"run{model}_{task}.sh"
        for model in ("deepsets", "mlp", "jetformer", "jedi_linear", "linformer")
        for task in ("train", "eval")
    )


def test_the_generator_names_the_rows_and_regimes_the_tests_do() -> None:
    # The generator keeps lists of its own, since it runs without the test suite; a
    # row added to one side and not the other would otherwise go unnoticed.
    assert tuple(make_train_catalogues.MODELS) == MODELS
    assert make_train_catalogues.EMBEDDED == set(EMBEDDED)
    assert make_train_catalogues.BASELINE_ONLY == set(BASELINE_ONLY)


def test_the_four_momentum_layout_names_no_feature_set_anywhere(
    tmp_path: Path,
) -> None:
    dbs = _empty_dbs(tmp_path)

    train = _render("train", dbs, Layout())
    evaluate = _render("eval", dbs, Layout())

    assert "data.features" not in train and "data.features" not in evaluate
    assert LABEL not in train and LABEL not in evaluate
    assert "hydra.run.dir=logs/train/hls4ml/mlp/nconst16/baseline" in train
    assert "hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/baseline" in evaluate


def test_every_eval_block_names_the_total_variation_once(tmp_path: Path) -> None:
    dbs = _empty_dbs(tmp_path)

    evaluate = _render("eval", dbs, Layout())

    assert evaluate.count("results_table.invariance_metric=tv") == len(GRID)


def test_best_trial_keeps_to_the_first_200_completed_trials(tmp_path: Path) -> None:
    # The overall best, 0.9, is trial 230, past the budget; within it 0.8 is shared by
    # trials 120 and 150, and the earlier one wins the tie.
    db = tmp_path / "x.db"
    peaks = {120: 0.8, 150: 0.8, 230: 0.9}
    _study(db, [peaks.get(number, 0.5) for number in range(250)])

    assert make_train_catalogues.best_trial(db, STUDY) == (120, 0.8, {"x": "0.8"})


def test_failed_and_running_trials_take_no_place_in_the_budget(tmp_path: Path) -> None:
    # A FAIL and an open RUNNING row before every 50th completed trial put the 200th
    # completed trial at number 207 and the 201st at 210. Counting every row would end
    # the window at number 199, where no trial beats 0.5.
    rows = []
    for k in range(201):
        rows += [TrialState.FAIL, TrialState.RUNNING] if k % 50 == 0 else []
        rows.append({199: 0.8, 200: 0.9}.get(k, 0.5))
    db = tmp_path / "x.db"
    _study(db, rows)

    assert make_train_catalogues.best_trial(db, STUDY) == (207, 0.8, {"x": "0.8"})


def test_best_trial_is_none_without_the_study_or_a_completed_trial(
    tmp_path: Path,
) -> None:
    db = tmp_path / "x.db"
    _study(db, [TrialState.FAIL, TrialState.RUNNING])

    assert make_train_catalogues.best_trial(db, STUDY) is None
    assert make_train_catalogues.best_trial(db, "absent") is None
