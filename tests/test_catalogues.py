"""The run catalogues and the submitters that read them.

The catalogues are documentation as much as they are input: every command that produced a published
number is a commented block in one of them. What has to hold is that the parser still finds the
whole grid of each model, that a train block and its evaluation block agree on where the manifest
is, and that the generator reproduces what is checked in. Only the grid is asked of all four
models: the parsing and the flags are the same code whichever catalogue is read.
"""

import subprocess
from functools import lru_cache
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


def _variants_of(model: str) -> tuple[str, ...]:
    """The regimes a model is catalogued under, as make_train_catalogues.py derives them.

    LorentzNet is the symmetry-embedded comparison class: it is trained plain and probed for the
    invariance it was built with, so it carries the baseline alone, 3 blocks against the 21 of the
    models that acquire their invariance from the data or the loss.
    """
    return VARIANTS[:1] if model == "lorentznet" else VARIANTS


def _blocks_of(model: str) -> int:
    return len(NCONSTITUENTS) * len(_variants_of(model))


@lru_cache
def _submit(task: str, catalogue: str, *flags: str) -> tuple[str, ...]:
    """Run one submitter in dry-run mode and return the commands it would have launched.

    Cached, because several tests read the same catalogue and each read is a subprocess.

    :param task: one of train, eval and search, naming both the submitter and the catalogue suffix.
    :raises AssertionError: with the submitter's stderr, when it exits non-zero.
    """
    submitter = f"scripts/cluster/submit_{task}.sh"
    argv = [*flags, f"scripts/hls4ml/run{catalogue}_{task}.sh"]
    result = subprocess.run(
        ["bash", submitter, "--dry-run", *argv], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    return tuple(line for line in result.stdout.splitlines() if line.startswith("["))


def _override(command: str, key: str) -> str:
    """The value of one Hydra override inside a command line."""
    for token in command.split():
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    raise AssertionError(f"{key} is absent from {command}")


@pytest.mark.parametrize("model", MODELS)
def test_every_catalogue_holds_the_whole_grid(model: str) -> None:
    """Three constituent counts by the model's own regimes, in the order the tables report."""
    expected = [
        f"hls4ml/{model}_{nconstituents}"
        for nconstituents in NCONSTITUENTS
        for _ in _variants_of(model)
    ]
    for task in ("train", "eval", "search"):
        commands = _submit(task, model)

        assert len(commands) == _blocks_of(model)
        assert [_override(command, "experiment") for command in commands] == expected


def test_a_train_block_names_its_regime_and_its_evaluation_reads_its_manifest() -> None:
    """The directory names are what --only matches, and the manifest path is the only thing joining
    the two generated catalogues."""
    directories = [_override(command, "hydra.run.dir") for command in _submit("train", "deepsets")]
    manifests = [
        _override(command, "checkpoint_manifest_path") for command in _submit("eval", "deepsets")
    ]

    assert directories == [
        f"logs/train/hls4ml/deepsets/nconst{nconstituents}/{variant}"
        for nconstituents in NCONSTITUENTS
        for variant in VARIANTS
    ]
    assert manifests == [f"{directory}/checkpoint_manifest.json" for directory in directories]


def test_a_search_block_names_the_right_space_and_stays_out_of_results() -> None:
    """The three symm_* regimes have one hyperparameter more than the other four, and a hundred
    trials with the full callbacks would fill results/ with untuned rows."""
    for command in _submit("search", "deepsets"):
        space, method = _override(command, "hparams_search"), _override(command, "data.method")

        assert space.endswith("_scl_optuna") == (method == "symm_loss")
        assert "callbacks=search" in command
        assert "logger=none" in command
        assert "data.n_folds=1" in command
    # LorentzNet is invariant by construction, so it has no SCL regime and no weight to search.
    assert not any(
        "symm_loss" in command or "_scl_optuna" in command
        for command in _submit("search", "lorentznet")
    )


def test_the_submitter_flags_reshape_every_command() -> None:
    """--cluster hands each block to submitit, --gpu pins a queue, --only selects a rerun."""
    # Submitit turns each block into a one-job multirun, whose output moves to <sweep.dir>/0, and
    # the manifest path must follow it there.
    train_command = _submit("train", "deepsets", "--cluster", "clariden")[0]
    eval_command = _submit("eval", "deepsets", "--cluster", "clariden")[0]

    assert "-m hydra/launcher=submitit_slurm_clariden" in train_command
    assert "hydra.launcher.timeout_min=" in train_command
    assert "hydra.sweep.dir=logs/train/hls4ml/deepsets/nconst16/baseline" in train_command
    assert "hydra.run.dir=" not in train_command
    assert _override(eval_command, "checkpoint_manifest_path").endswith(
        "/nconst16/baseline/0/checkpoint_manifest.json"
    )

    # Four queues on one host stay independent only if each pins a GPU of its own.
    assert all(
        "trainer.devices=[3]" in command for command in _submit("train", "deepsets", "--gpu", "3")
    )

    # The labels are how a rerun of one regime is asked for after a failure.
    assert len(_submit("train", "deepsets", "--only", "symm_so2")) == len(NCONSTITUENTS)
    assert len(_submit("train", "deepsets", "--only", "nconst150")) == len(VARIANTS)
    assert len(_submit("train", "deepsets", "--only", "1,2")) == 2


def test_the_checked_in_catalogues_are_what_the_generator_writes(tmp_path: Path) -> None:
    """They are generated, not edited, so a hand-made change must be caught here.

    With no study databases every block is the transferred one, which is the state the catalogues
    are checked in at, before any search has run.
    """
    generator = ["python", "scripts/optuna/make_train_catalogues.py", "--dry-run"]
    result = subprocess.run(
        [*generator, "--dbs", str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WOULD CHANGE" not in result.stdout
    for model in MODELS:
        for task in ("train", "eval"):
            assert f"run{model}_{task}.sh: {_blocks_of(model)} commands" in result.stdout
