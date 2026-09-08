import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import ROOT, dry_run

LAUNCHER = "scripts/cluster/submit_search_ngt.sh"
SUBMITTER = "scripts/cluster/submit_search.sh"
CATALOGUE = "scripts/hls4ml/runlorentznet_search.sh"
# One study, four workers over three MIG slices: the shape every assertion below is written for.
POOL = ("--only", "nconst150", "--workers", "4", "--trials", "10", "--seed", "7", "--tag", "t")
SLICES = "MIG-aaaa,MIG-bbbb,MIG-cccc"

# scripts/ is not a package, so the module is loaded by path.
_SPEC = importlib.util.spec_from_file_location(
    "sweep_status", ROOT / "scripts" / "cluster" / "sweep_status.py"
)
sweep_status = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep_status)


def _launch(*argv: str, devices: str = "") -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "NVIDIA_VISIBLE_DEVICES"}
    if devices:
        env["NVIDIA_VISIBLE_DEVICES"] = devices
    return subprocess.run(
        ["bash", LAUNCHER, *argv], cwd=ROOT, capture_output=True, text=True, env=env
    )


def _worker_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]


def _value(line: str, key: str) -> str:
    tokens = line.split()
    if key.startswith("--"):
        return tokens[tokens.index(key) + 1]
    return next(token for token in tokens if token.startswith(f"{key}=")).split("=", 1)[1]


def _without_sweep_dir(command: str) -> str:
    return " ".join(t for t in command.split() if not t.startswith("hydra.sweep.dir="))


def test_the_pool_shares_out_the_devices_the_budget_and_the_seeds() -> None:
    result = _launch("--dry-run", *POOL, CATALOGUE, "trainer.max_epochs=3", devices=SLICES)
    assert result.returncode == 0, result.stderr
    lines = _worker_lines(result.stdout)

    assert len(lines) == 4
    # Devices repeat round-robin; the seeds never do, or the workers would search the same points.
    assert [_value(line, "CUDA_VISIBLE_DEVICES") for line in lines] == [
        "MIG-aaaa",
        "MIG-bbbb",
        "MIG-cccc",
        "MIG-aaaa",
    ]
    assert [_value(line, "hydra.sweeper.n_trials") for line in lines] == ["3", "3", "2", "2"]
    assert [_value(line, "hydra.sweeper.sampler.seed") for line in lines] == ["7", "8", "9", "10"]
    assert [_value(line, "--tag") for line in lines] == [f"t_w{j}" for j in range(4)]
    assert [line.split(" > ")[1].split()[0] for line in lines] == [
        f"logs/searches/ngt/t_w{j}.log" for j in range(4)
    ]
    assert all("trainer.max_epochs=3" in line for line in lines)
    assert all(_value(line, "--only") == "nconst150" for line in lines)
    # Off the pod there is no cpus command, so nothing is pinned.
    assert not any("taskset" in line for line in lines)


def test_the_workers_are_pinned_to_contiguous_shares_of_the_given_cpus() -> None:
    # Five ids, given as the pods print them, over four workers: the first worker takes the odd one.
    result = _launch("--dry-run", *POOL, "--cpus", "64-66, 96,97", CATALOGUE, devices=SLICES)
    assert result.returncode == 0, result.stderr
    lines = _worker_lines(result.stdout)

    assert [line.split()[1:4] for line in lines] == [
        ["taskset", "-c", "64,65"],
        ["taskset", "-c", "66"],
        ["taskset", "-c", "96"],
        ["taskset", "-c", "97"],
    ]


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(("--trials", "10", "--seed", "7"), id="no --only"),
        pytest.param(("--only", "nconst150", "--seed", "7"), id="no --trials"),
        pytest.param(("--only", "nconst150", "--trials", "10"), id="no --seed"),
        pytest.param(
            ("--only", "nconst150", "--trials", "10", "--seed", "7", "--workers", "0"),
            id="no workers",
        ),
        pytest.param(
            ("--only", "nconst150", "--trials", "2", "--seed", "7", "--workers", "3"),
            id="more workers than trials",
        ),
        pytest.param(
            ("--only", "nconst150", "--trials", "10", "--seed", "7", "--gpus", "0,3"),
            id="device past the list",
        ),
        pytest.param(
            ("--only", "nconst150", "--trials", "10", "--seed", "7", "--cpus", "64-66,all"),
            id="cpu list that is not ids",
        ),
        pytest.param(
            ("--only", "nconst150", "--trials", "10", "--seed", "7", "--cpus", "64,65"),
            id="fewer cpus than workers",
        ),
    ],
)
def test_a_pool_that_cannot_be_shared_out_is_refused(flags: tuple[str, ...]) -> None:
    assert _launch("--dry-run", *flags, CATALOGUE, devices=SLICES).returncode != 0


def test_a_real_launch_without_devices_is_refused() -> None:
    result = _launch("--only", "nconst150", "--trials", "10", "--seed", "7", CATALOGUE)

    assert result.returncode != 0
    assert "NVIDIA_VISIBLE_DEVICES" in result.stderr


def test_a_tag_moves_the_sweep_directory_and_nothing_else() -> None:
    plain = dry_run("search", CATALOGUE)
    tagged = dry_run("search", CATALOGUE, "--tag", "w0")

    assert [_without_sweep_dir(c) for c in tagged] == [_without_sweep_dir(c) for c in plain]
    assert all(_value(c, "hydra.sweep.dir").endswith("_w0") for c in tagged)
    assert not any(_value(c, "hydra.sweep.dir").endswith("_w0") for c in plain)


def test_a_tagged_worker_leaves_another_workers_live_log_alone(tmp_path: Path) -> None:
    # A throwaway root holding the submitter and one block that names no hparams_search, so no
    # study database is created; with PYTHON=echo the driver writes its command line and exits.
    (tmp_path / ".project-root").touch()
    (tmp_path / "scripts" / "hls4ml").mkdir(parents=True)
    shutil.copytree(ROOT / "scripts" / "cluster", tmp_path / "scripts" / "cluster")
    (tmp_path / "scripts" / "hls4ml" / "runfake_search.sh").write_text(
        "# python3 src/train.py -m experiment=hls4ml/lorentznet_16 \\\n"
        "#     data.method=augmentation data.augmentation_scheme=none\n"
    )
    logdir = tmp_path / "logs" / "searches" / "hls4ml_runfake_search"
    logdir.mkdir(parents=True)
    live = logdir / "1_lorentznet_nconst16_baseline.log"
    live.write_text("a worker of another pod is writing here\n")

    subprocess.run(
        ["bash", SUBMITTER, "--tag", "w1", "scripts/hls4ml/runfake_search.sh"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHON": "echo", "HF_HUB_CACHE": str(tmp_path / "hf")},
    )

    # On the pods logs/ is one shared volume, so rotating the live log would orphan its writer.
    assert live.read_text() == "a worker of another pod is writing here\n"
    assert list(logdir.glob("*.log.*")) == []
    assert "src/train.py" in (logdir / "1_lorentznet_nconst16_baseline_w1.log").read_text()


def test_a_study_is_alive_while_any_of_its_workers_is(tmp_path: Path) -> None:
    (tmp_path / "1_lbl.log").write_text("Traceback (most recent call last)\n")
    (tmp_path / "1_lbl_s1_w1.log").write_text("Epoch 3\n")
    # A copy a restart moved aside is history, not a worker.
    (tmp_path / "1_lbl.log.20260101-000000").write_text("Epoch 1\n")

    logs = sweep_status.worker_logs(str(tmp_path), 1, "lbl")

    assert [log.name for log in logs] == ["1_lbl.log", "1_lbl_s1_w1.log"]
    assert sweep_status.classify_workers(logs)[0] == "alive"
    assert sweep_status.classify_workers([])[0] == "no-log"


def test_with_no_worker_alive_the_newest_log_carries_the_verdict(tmp_path: Path) -> None:
    dead = tmp_path / "1_lbl_s1_w0.log"
    dead.write_text("Traceback (most recent call last)\n")
    (tmp_path / "1_lbl_s1_w1.log").write_text("[2026-01-01][HYDRA] Best value: 0.8\n")
    os.utime(dead, (0, 0))

    logs = sweep_status.worker_logs(str(tmp_path), 1, "lbl")

    assert sweep_status.classify_workers(logs)[0] == "FINISHED"
