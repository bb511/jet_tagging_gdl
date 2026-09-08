import subprocess
from pathlib import Path

import pytest

from tests.conftest import MODELS, NCONSTITUENTS, ROOT, dry_run

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
    # LorentzNet, the symmetry-embedded class, is trained in the baseline regime alone.
    return VARIANTS[:1] if model == "lorentznet" else VARIANTS


def _blocks_of(model: str) -> int:
    return len(NCONSTITUENTS) * len(_variants_of(model))


def _submit(task: str, model: str, *flags: str) -> tuple[str, ...]:
    return dry_run(task, f"scripts/hls4ml/run{model}_{task}.sh", *flags)


def _field(command: str, key: str) -> str:
    result = subprocess.run(
        ["bash", "-c", '. scripts/cluster/lib.sh; field "$1" "$2"', "_", command, key],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _override(command: str, key: str) -> str:
    # Hydra keeps the last value of a repeated override, which is how a launcher beats a block.
    values = [t.split("=", 1)[1] for t in command.split() if t.startswith(f"{key}=")]
    assert values, f"{key} is absent from {command}"
    return values[-1]


def _has_override(command: str, key: str) -> bool:
    return any(token.startswith(f"{key}=") for token in command.split())


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # What the generator writes now, not what is checked in: the catalogues are regenerated on
    # the machine holding the studies, so they lag a change to the generator.
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


@pytest.mark.parametrize("model", MODELS)
def test_every_catalogue_holds_the_whole_grid(model: str) -> None:
    expected = [f"hls4ml/{model}_{n}" for n in NCONSTITUENTS for _ in _variants_of(model)]
    for task in ("train", "eval", "search"):
        commands = _submit(task, model)

        assert len(commands) == _blocks_of(model)
        assert [_override(command, "experiment") for command in commands] == expected


def test_a_train_block_names_its_regime_and_its_evaluation_reads_its_manifest() -> None:
    directories = [_override(c, "hydra.run.dir") for c in _submit("train", "deepsets")]
    manifests = [_override(c, "checkpoint_manifest_path") for c in _submit("eval", "deepsets")]

    assert directories == [
        f"logs/train/hls4ml/deepsets/nconst{n}/{variant}"
        for n in NCONSTITUENTS
        for variant in VARIANTS
    ]
    assert manifests == [f"{directory}/checkpoint_manifest.json" for directory in directories]


def test_a_search_block_names_the_right_space_and_stays_out_of_results() -> None:
    for command in _submit("search", "deepsets"):
        space, method = _override(command, "hparams_search"), _override(command, "data.method")

        assert space.endswith("_scl_optuna") == (method == "symm_loss")
        assert "callbacks=search" in command
        assert "logger=none" in command
        assert "data.n_folds=1" in command
    assert not any("symm_loss" in c or "_scl_optuna" in c for c in _submit("search", "lorentznet"))


def test_the_submitter_flags_reshape_every_command() -> None:
    # Submitit runs a block as a one-job multirun writing to <sweep.dir>/0, and the manifest path
    # has to follow the checkpoints there.
    train_command = _submit("train", "deepsets", "--cluster", "clariden")[0]
    eval_command = _submit("eval", "deepsets", "--cluster", "clariden")[0]

    assert "-m hydra/launcher=submitit_slurm_clariden" in train_command
    assert "hydra.launcher.timeout_min=" in train_command
    assert "hydra.sweep.dir=logs/train/hls4ml/deepsets/nconst16/baseline" in train_command
    assert "hydra.run.dir=" not in train_command
    assert _override(eval_command, "checkpoint_manifest_path").endswith(
        "/nconst16/baseline/0/checkpoint_manifest.json"
    )
    assert all("trainer.devices=[3]" in c for c in _submit("train", "deepsets", "--gpu", "3"))
    assert len(_submit("train", "deepsets", "--only", "symm_so2")) == len(NCONSTITUENTS)
    assert len(_submit("train", "deepsets", "--only", "nconst150")) == len(VARIANTS)
    assert len(_submit("train", "deepsets", "--only", "1,2")) == 2


def test_folds_fan_a_block_out_into_a_main_run_and_one_process_per_fold() -> None:
    lines = _submit("train", "deepsets", "--only", "1,2", "--folds", "2", "--gpus", "2,3")
    base = "logs/train/hls4ml/deepsets/nconst16"

    assert len(lines) == 6
    assert [line.split()[1] for line in lines] == ["main:", "fold_1:", "fold_2:"] * 2
    assert [_override(line, "data.n_folds") for line in lines] == ["1", "2", "2"] * 2
    assert not any(_has_override(line, "fold") for line in lines[::3])
    assert [_override(line, "fold") for line in lines[1:3]] == ["1", "2"]
    assert [_override(line, "hydra.run.dir") for line in lines[:3]] == [
        f"{base}/baseline",
        f"{base}/baseline/fold_1",
        f"{base}/baseline/fold_2",
    ]
    assert _override(lines[3], "hydra.run.dir") == f"{base}/aug_so2"
    # The device counter runs on across blocks, so two blocks over two devices load them evenly.
    assert [_override(line, "trainer.devices") for line in lines] == ["[2]", "[3]"] * 3
    assert not any(_has_override(line, "CUDA_VISIBLE_DEVICES") for line in lines)
    # The block's runner expands $parent off the main run's manifest, so the folds nest under it.
    assert not any(_has_override(line, "parent_run_id") for line in lines[::3])
    assert [_override(line, "parent_run_id") for line in lines[1:3]] == ["$parent"] * 2


def test_a_uuid_device_is_pinned_through_the_environment() -> None:
    uuids = "MIG-aaaa,GPU-bbbb,MIG-cccc"
    lines = _submit("train", "deepsets", "--only", "1", "--folds", "2", "--gpus", uuids)

    # CUDA cannot index a UUID, so it goes into the environment and trainer.devices stays at 0.
    assert [_override(line, "CUDA_VISIBLE_DEVICES") for line in lines] == uuids.split(",")
    assert all(": CUDA_VISIBLE_DEVICES=" in line for line in lines)
    assert all(_override(line, "trainer.devices") == "[0]" for line in lines)


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
    result = subprocess.run(
        ["bash", "scripts/cluster/submit_train.sh", "--dry-run", *flags, TRAIN_CATALOGUE],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_the_checked_in_catalogues_are_what_the_generator_writes(tmp_path: Path) -> None:
    generator = ["python", "scripts/optuna/make_train_catalogues.py", "--dry-run"]
    result = subprocess.run(
        [*generator, "--dbs", str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )

    assert "error" not in result.stderr, result.stderr
    for model in MODELS:
        for task in ("train", "eval"):
            # A catalogue holding best trials is reproducible only with the databases that chose
            # them, so of it only the block count is checked.
            tuned = "# best trial" in (ROOT / f"scripts/hls4ml/run{model}_{task}.sh").read_text()
            line = f"run{model}_{task}.sh: {_blocks_of(model)} commands"
            assert (line if tuned else f"{line}, unchanged") in result.stdout, result.stdout


def test_an_eval_block_names_its_baseline_unless_it_is_one(generated: Path) -> None:
    commands = dry_run("eval", str(generated / "rundeepsets_eval.sh"))
    trained = [c for c in commands if _override(c, "data.augmentation_scheme") != "none"]
    baselines = [c for c in commands if c not in trained]
    lorentznet = dry_run("eval", str(generated / "runlorentznet_eval.sh"))

    assert len(trained) == len(NCONSTITUENTS) * (len(VARIANTS) - 1)
    assert [_override(c, "baseline_checkpoint_manifest_path") for c in trained] == [
        f"logs/train/hls4ml/deepsets/nconst{n}/baseline/checkpoint_manifest.json"
        for n in NCONSTITUENTS
        for _ in VARIANTS[1:]
    ]
    # Adjacent, so the pair reads as one thing: what to test, and what to test it against.
    for command in trained:
        tokens = command.split()
        own = next(i for i, t in enumerate(tokens) if t.startswith("checkpoint_manifest_path="))
        assert tokens[own + 1].startswith("baseline_checkpoint_manifest_path=")
    # A baseline is its own comparison, and LorentzNet is trained in the baseline regime alone.
    assert len(baselines) == len(NCONSTITUENTS) and len(lorentznet) == len(NCONSTITUENTS)
    for command in [*baselines, *lorentznet]:
        assert not _has_override(command, "baseline_checkpoint_manifest_path")


def test_an_override_key_is_read_whole_and_not_as_the_tail_of_a_longer_one() -> None:
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
    command = dry_run("eval", str(generated / "rundeepsets_eval.sh"), "--cluster", "clariden")[1]

    assert _override(command, "data.augmentation_scheme") == "so2"
    assert _override(command, "checkpoint_manifest_path").endswith(
        "/nconst16/aug_so2/0/checkpoint_manifest.json"
    )
    assert _override(command, "baseline_checkpoint_manifest_path").endswith(
        "/nconst16/baseline/0/checkpoint_manifest.json"
    )
