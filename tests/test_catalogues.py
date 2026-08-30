"""The run catalogues and the submitters that read them.

The catalogues are documentation as much as they are input: every command that produced a published
number is a commented block in one of them. What has to hold is that the parser still finds all 21
blocks of each grid, that a train block and its evaluation block agree on where the manifest is,
and that the generator reproduces what is checked in.
"""

import subprocess
from pathlib import Path

import pytest

from tests.conftest import ROOT

MODELS = ("deepsets", "mlp", "jetformer", "lorentznet")
NCONSTITUENTS = (16, 50, 150)
VARIANTS = (
    "baseline",
    "aug_so2",
    "aug_so3",
    "aug_lorentz",
    "symm_so2",
    "symm_so3",
    "symm_lorentz",
)
BLOCKS_PER_CATALOGUE = len(NCONSTITUENTS) * len(VARIANTS)


def _submit(task: str, catalogue: str, *flags: str) -> list[str]:
    """Run one submitter in dry-run mode and return the commands it would have launched.

    :param task: one of train, eval and search, naming both the submitter and the catalogue suffix.
    :raises AssertionError: with the submitter's stderr, when it exits non-zero.
    """
    result = subprocess.run(
        [
            "bash",
            f"scripts/cluster/submit_{task}.sh",
            "--dry-run",
            *flags,
            f"scripts/hls4ml/run{catalogue}_{task}.sh",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    return [line for line in result.stdout.splitlines() if line.startswith("[")]


def _override(command: str, key: str) -> str:
    """The value of one Hydra override inside a command line."""
    for token in command.split():
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    raise AssertionError(f"{key} is absent from {command}")


@pytest.mark.parametrize("task", ["train", "eval", "search"])
@pytest.mark.parametrize("model", MODELS)
def test_every_catalogue_holds_the_whole_grid(task: str, model: str) -> None:
    """Three constituent counts by seven regimes, in the order every table reports them."""
    commands = _submit(task, model)

    assert len(commands) == BLOCKS_PER_CATALOGUE
    expected = [
        f"hls4ml/{model}_{nconstituents}" for nconstituents in NCONSTITUENTS for _ in VARIANTS
    ]
    assert [_override(command, "experiment") for command in commands] == expected


@pytest.mark.parametrize("model", MODELS)
def test_the_run_directories_name_the_regime_of_each_block(model: str) -> None:
    """The directory names are what --only matches and what the results tables are read by."""
    commands = _submit("train", model)
    directories = [_override(command, "hydra.run.dir") for command in commands]

    assert directories == [
        f"logs/train/hls4ml/{model}/nconst{nconstituents}/{variant}"
        for nconstituents in NCONSTITUENTS
        for variant in VARIANTS
    ]


@pytest.mark.parametrize("model", MODELS)
def test_each_evaluation_reads_the_manifest_its_training_wrote(model: str) -> None:
    """The two catalogues are generated together, and this is the only thing joining them."""
    train_dirs = [_override(command, "hydra.run.dir") for command in _submit("train", model)]
    manifests = [
        _override(command, "checkpoint_manifest_path") for command in _submit("eval", model)
    ]

    assert manifests == [f"{directory}/checkpoint_manifest.json" for directory in train_dirs]


@pytest.mark.parametrize("model", MODELS)
def test_the_scl_blocks_search_the_space_that_carries_the_scl_weight(model: str) -> None:
    """The three symm_* regimes have one hyperparameter more than the other four."""
    commands = _submit("search", model)
    spaces = [_override(command, "hparams_search") for command in commands]
    methods = [_override(command, "data.method") for command in commands]

    assert all(
        space.endswith("_scl_optuna") == (method == "symm_loss")
        for space, method in zip(spaces, methods)
    )
    assert methods.count("symm_loss") == len(NCONSTITUENTS) * 3


@pytest.mark.parametrize("model", MODELS)
def test_a_search_trial_writes_neither_a_plot_nor_a_results_row(model: str) -> None:
    """A hundred trials with the full callbacks would fill results/ with untuned rows."""
    for command in _submit("search", model):
        assert "callbacks=search" in command
        assert "logger=none" in command
        assert "data.n_folds=1" in command


def test_a_cluster_run_gains_a_launcher_and_a_sweep_directory() -> None:
    """Submitit turns each block into a one-job multirun, which moves its output dir."""
    command = _submit("train", "deepsets", "--cluster", "clariden")[0]

    assert "-m hydra/launcher=submitit_slurm_clariden" in command
    assert "hydra.launcher.timeout_min=" in command
    assert "hydra.sweep.dir=logs/train/hls4ml/deepsets/nconst16/baseline" in command
    assert "hydra.run.dir=" not in command


def test_a_cluster_evaluation_follows_the_manifest_into_the_job_directory() -> None:
    """Hydra writes a multirun job to <sweep.dir>/0, and the manifest goes there with it."""
    command = _submit("eval", "deepsets", "--cluster", "clariden")[0]

    assert _override(command, "checkpoint_manifest_path").endswith(
        "/nconst16/baseline/0/checkpoint_manifest.json"
    )


def test_a_gpu_selection_reaches_every_command() -> None:
    """Four queues on one host stay independent only if each pins a GPU of its own."""
    commands = _submit("train", "deepsets", "--gpu", "3")

    assert all("trainer.devices=[3]" in command for command in commands)


def test_only_selects_by_label_and_by_index() -> None:
    """The labels are how a rerun of one regime is asked for after a failure."""
    assert len(_submit("train", "deepsets", "--only", "symm_so2")) == len(NCONSTITUENTS)
    assert len(_submit("train", "deepsets", "--only", "nconst150")) == len(VARIANTS)
    assert len(_submit("train", "deepsets", "--only", "1,2")) == 2


def test_the_checked_in_catalogues_are_what_the_generator_writes(tmp_path: Path) -> None:
    """They are generated, not edited, so a hand-made change must be caught here.

    With no study databases every block is the transferred one, which is the state the catalogues
    are checked in at, before any search has run.
    """
    result = subprocess.run(
        [
            "python",
            "scripts/optuna/make_train_catalogues.py",
            "--dry-run",
            "--dbs",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WOULD CHANGE" not in result.stdout
    assert result.stdout.count(f"{BLOCKS_PER_CATALOGUE} commands") == 2 * len(MODELS)
