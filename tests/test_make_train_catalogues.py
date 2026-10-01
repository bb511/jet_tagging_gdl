import importlib.util
import re
import sys
from pathlib import Path

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


def test_a_baseline_only_comparator_has_its_three_blocks_in_every_layout(
    tmp_path: Path,
) -> None:
    # JEDI-linear runs the baseline alone, as the embedded class does, but reads
    # data.features, so every feature set has its catalogues too; the header gives the
    # comparator's reason for the single regime rather than the embedded class's.
    dbs = _empty_dbs(tmp_path)

    for layout in (Layout(), Layout(features=FEATURES)):
        assert "jedi_linear" in make_train_catalogues.models_of(layout)
        for task in ("train", "eval"):
            text = make_train_catalogues.render("jedi_linear", task, dbs, layout)
            assert make_train_catalogues.count_commands(LIB, text) == 3
            assert text.count("data.augmentation_scheme=none") == 3
            assert "data.method=symm_loss" not in text
            assert text.count(OVERRIDE) == (3 if layout.features else 0)
            assert "comparator from the literature" in text
            assert "embeds the symmetry" not in text
    # The phrase checked above is live: the embedded class's catalogues carry it.
    vn_z = make_train_catalogues.render("vn_z", "train", dbs, Layout())
    assert "embeds the symmetry" in vn_z


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


def test_the_transferred_ablation_of_a_feature_set_carries_both_marks(
    tmp_path: Path,
) -> None:
    dbs = _empty_dbs(tmp_path)
    layout = Layout(features=FEATURES, transferred=True)

    train = _render("train", dbs, layout)
    evaluate = _render("eval", dbs, layout)

    run = f"logs/train/hls4ml/{LABEL}/mlp/nconst16/baseline_transferred"
    assert f"hydra.run.dir={run}" in train
    assert run in _manifest_paths(evaluate)
    assert train.count(OVERRIDE) == len(GRID)
    assert evaluate.count(OVERRIDE) == len(GRID)
