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
TRAIN_CATALOGUE = "scripts/hls4ml/rundeepsets_train.sh"
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
def _dry_run(task: str, catalogue: str, *flags: str) -> tuple[str, ...]:
    """Run one submitter in dry-run mode and return the commands it would have launched.

    Cached, because several tests read the same catalogue and each read is a subprocess.

    :param task: one of train, eval and search, naming the submitter.
    :param catalogue: path to the catalogue file, relative to the repository root or absolute.
    :raises AssertionError: with the submitter's stderr, when it exits non-zero.
    """
    result = subprocess.run(
        ["bash", f"scripts/cluster/submit_{task}.sh", "--dry-run", *flags, catalogue],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    return tuple(line for line in result.stdout.splitlines() if line.startswith("["))


def _submit(task: str, catalogue: str, *flags: str) -> tuple[str, ...]:
    """The same, for one of the checked-in catalogues, named by its model."""
    return _dry_run(task, f"scripts/hls4ml/run{catalogue}_{task}.sh", *flags)


def _field(command: str, key: str) -> str:
    """What lib.sh's own override reader makes of ``key`` in ``command``."""
    result = subprocess.run(
        ["bash", "-c", '. scripts/cluster/lib.sh; field "$1" "$2"', "_", command, key],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    return result.stdout.strip()


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh generation of every catalogue, with no study database to draw from.

    The files under scripts/hls4ml/ are regenerated on the machine that holds the studies, so a
    change to the generator reaches them later than it reaches this test: what the generator
    writes now is read here, not what is checked in.
    """
    out = tmp_path_factory.mktemp("catalogues")
    subprocess.run(
        ["python", "scripts/optuna/make_train_catalogues.py"]
        + ["--out", str(out), "--dbs", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out


def _override(command: str, key: str) -> str:
    """The value of one Hydra override inside a command line."""
    for token in command.split():
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    raise AssertionError(f"{key} is absent from {command}")


def _last_override(command: str, key: str) -> str:
    """The value Hydra keeps of an override a launcher appended behind the block's own: the
    last."""
    values = [t.split("=", 1)[1] for t in command.split() if t.startswith(f"{key}=")]
    assert values, f"{key} is absent from {command}"

    return values[-1]


def _has_override(command: str, key: str) -> bool:
    return any(token.startswith(f"{key}=") for token in command.split())


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
    """The three symm_* regimes have one hyperparameter more than the other four, and two hundred
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


def test_folds_fan_a_block_out_into_a_main_run_and_one_process_per_fold() -> None:
    """--folds k is k+1 detached processes per block, each on the next device of --gpus.

    The main run trains fold-less and fold j alone under <run dir>/fold_<j>, the layout
    merge_manifests reads back; the launcher's overrides come last, so they beat the block's
    data.n_folds=5. The device counter runs on across blocks, so two blocks over two devices load
    them evenly.
    """
    lines = _submit("train", "deepsets", "--only", "1,2", "--folds", "2", "--gpus", "2,3")
    base = "logs/train/hls4ml/deepsets/nconst16"

    assert len(lines) == 6
    assert [line.split()[1] for line in lines] == ["main:", "fold_1:", "fold_2:"] * 2
    assert [_last_override(line, "data.n_folds") for line in lines] == ["1", "2", "2"] * 2
    assert not any(_has_override(line, "fold") for line in lines[::3])
    assert [_last_override(line, "fold") for line in lines[1:3]] == ["1", "2"]
    assert [_last_override(line, "hydra.run.dir") for line in lines[:3]] == [
        f"{base}/baseline",
        f"{base}/baseline/fold_1",
        f"{base}/baseline/fold_2",
    ]
    assert _last_override(lines[3], "hydra.run.dir") == f"{base}/aug_so2"
    assert [_last_override(line, "trainer.devices") for line in lines] == ["[2]", "[3]"] * 3
    assert not any(_has_override(line, "CUDA_VISIBLE_DEVICES") for line in lines)


def test_a_uuid_device_is_pinned_through_the_environment() -> None:
    """A MIG slice or a full card on an NGT pod is a UUID CUDA cannot index, so it goes into
    CUDA_VISIBLE_DEVICES ahead of the command and trainer.devices stays at 0, as
    submit_search_ngt.sh does."""
    uuids = "MIG-aaaa,GPU-bbbb,MIG-cccc"
    lines = _submit("train", "deepsets", "--only", "1", "--folds", "2", "--gpus", uuids)

    assert [_last_override(line, "CUDA_VISIBLE_DEVICES") for line in lines] == uuids.split(",")
    assert all(": CUDA_VISIBLE_DEVICES=" in line for line in lines)
    assert all(_last_override(line, "trainer.devices") == "[0]" for line in lines)


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(("--folds", "2"), id="no --gpus"),
        pytest.param(
            ("--folds", "2", "--gpus", "0", "--cluster", "clariden"), id="with --cluster"
        ),
        pytest.param(("--folds", "1", "--gpus", "0"), id="one fold"),
        pytest.param(("--folds", "2", "--gpus", "0,cuda:1"), id="unplaceable device"),
    ],
)
def test_a_fold_launch_that_cannot_be_placed_is_refused(flags: tuple[str, ...]) -> None:
    """Each would start processes sharing a GPU blindly, a fold train.py rejects, or a slurm job
    per fold on top of the one per block."""
    result = subprocess.run(
        ["bash", "scripts/cluster/submit_train.sh", "--dry-run", *flags, TRAIN_CATALOGUE],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_the_checked_in_catalogues_are_what_the_generator_writes(tmp_path: Path) -> None:
    """They are generated, not edited, so a hand-made change must be caught here.

    With no study databases every block is the transferred one, which is the state a catalogue is
    checked in at before its searches have run. One carrying the best trials can only be reproduced
    with the databases that chose them, so of it only the block count is checked.
    """
    generator = ["python", "scripts/optuna/make_train_catalogues.py", "--dry-run"]
    result = subprocess.run(
        [*generator, "--dbs", str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )

    assert "error" not in result.stderr, result.stderr
    for model in MODELS:
        for task in ("train", "eval"):
            tuned = (
                "# best trial" in (ROOT / "scripts/hls4ml" / f"run{model}_{task}.sh").read_text()
            )
            line = f"run{model}_{task}.sh: {_blocks_of(model)} commands"
            assert (line if tuned else f"{line}, unchanged") in result.stdout, result.stdout


def test_a_non_baseline_eval_block_names_its_baseline(generated: Path) -> None:
    """The delta, confidence-interval and p-value columns come from running the baseline of the
    same architecture and constituent count on the same transformed batches, so a block has to say
    where that run's manifest is, and has to say it beside its own."""
    commands = _dry_run("eval", str(generated / "rundeepsets_eval.sh"))
    trained = [c for c in commands if _override(c, "data.augmentation_scheme") != "none"]

    assert len(trained) == len(NCONSTITUENTS) * (len(VARIANTS) - 1)
    assert [_override(command, "baseline_checkpoint_manifest_path") for command in trained] == [
        f"logs/train/hls4ml/deepsets/nconst{nconstituents}/baseline/checkpoint_manifest.json"
        for nconstituents in NCONSTITUENTS
        for _ in VARIANTS[1:]
    ]
    # Adjacent, because the pair reads as one thing: what to test, and what to test it against.
    for command in trained:
        tokens = command.split()
        own = next(i for i, t in enumerate(tokens) if t.startswith("checkpoint_manifest_path="))

        assert tokens[own + 1].startswith("baseline_checkpoint_manifest_path=")


def test_a_run_with_no_baseline_to_compare_with_names_none(generated: Path) -> None:
    """A baseline block is its own comparison, and LorentzNet, the symmetry-embedded class, is
    trained in the baseline regime alone, so neither is scored against another run."""
    baselines = [
        command
        for command in _dry_run("eval", str(generated / "rundeepsets_eval.sh"))
        if _override(command, "data.augmentation_scheme") == "none"
    ]
    lorentznet = _dry_run("eval", str(generated / "runlorentznet_eval.sh"))

    assert len(baselines) == len(NCONSTITUENTS)
    assert len(lorentznet) == len(NCONSTITUENTS)
    for command in [*baselines, *lorentznet]:
        assert not _has_override(command, "baseline_checkpoint_manifest_path")


def test_an_override_key_is_read_whole_and_not_as_the_tail_of_a_longer_one() -> None:
    """checkpoint_manifest_path is a suffix of baseline_checkpoint_manifest_path, so an unanchored
    match would hand the submitters whichever of the two came first in the block."""
    own = "checkpoint_manifest_path=logs/own/checkpoint_manifest.json"
    base = "baseline_checkpoint_manifest_path=logs/base/checkpoint_manifest.json"

    for order in (f"src/eval.py {own} {base}", f"src/eval.py {base} {own}"):
        assert _field(order, "checkpoint_manifest_path") == "logs/own/checkpoint_manifest.json"
        assert (
            _field(order, "baseline_checkpoint_manifest_path")
            == "logs/base/checkpoint_manifest.json"
        )
    assert _field(f"src/eval.py {own}", r"data\.method") == ""


def test_cluster_mode_moves_both_manifest_paths_into_the_multirun_job_directory(
    generated: Path,
) -> None:
    """Submitit runs a block as a one-job multirun writing to <sweep.dir>/0, and the block is
    useless if only one of its two manifests follows the checkpoints there."""
    command = _dry_run("eval", str(generated / "rundeepsets_eval.sh"), "--cluster", "clariden")[1]

    assert _override(command, "data.augmentation_scheme") == "so2"
    assert _override(command, "checkpoint_manifest_path").endswith(
        "/nconst16/aug_so2/0/checkpoint_manifest.json"
    )
    assert _override(command, "baseline_checkpoint_manifest_path").endswith(
        "/nconst16/baseline/0/checkpoint_manifest.json"
    )
