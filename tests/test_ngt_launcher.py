"""Many workers over one Optuna study, and the bookkeeping that keeps them apart.

The sweeper opens its study with ``load_if_exists``, so a pool of drivers sharing a study name is a
matter of bookkeeping rather than of the search itself: each worker needs a seed of its own, a
slice of the trial budget, a device and a log no sibling rotates. That bookkeeping is what is
tested here, never a sweep. Everything runs under ``--dry-run``, except one launch against a
throwaway tree holding a copy of the submitter and a two-line catalogue, whose driver is ``echo``.
"""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import ROOT

LAUNCHER = "scripts/cluster/submit_search_ngt.sh"
SUBMITTER = "scripts/cluster/submit_search.sh"
CATALOGUE = "scripts/hls4ml/runlorentznet_search.sh"

# One study, four workers over three MIG slices: the shape every assertion below is written for.
POOL = ("--only", "nconst150", "--workers", "4", "--trials", "10", "--seed", "7", "--tag", "t")
SLICES = "MIG-aaaa,MIG-bbbb,MIG-cccc"


def _load_sweep_status() -> ModuleType:
    """Import ``scripts/cluster/sweep_status.py`` by path; ``scripts/`` is not a package."""
    path = ROOT / "scripts" / "cluster" / "sweep_status.py"
    spec = importlib.util.spec_from_file_location("sweep_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


sweep_status = _load_sweep_status()


def _launch(*argv: str, devices: str = "") -> subprocess.CompletedProcess[str]:
    """Run the NGT launcher with ``NVIDIA_VISIBLE_DEVICES`` set to *devices*, or unset when
    empty."""
    env = {key: value for key, value in os.environ.items() if key != "NVIDIA_VISIBLE_DEVICES"}
    if devices:
        env["NVIDIA_VISIBLE_DEVICES"] = devices

    return subprocess.run(
        ["bash", LAUNCHER, *argv], cwd=ROOT, capture_output=True, text=True, env=env
    )


def _worker_lines(stdout: str) -> list[str]:
    """One line per worker: the environment prefix and the command the launcher would run."""
    return [line for line in stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]


def _value(line: str, key: str) -> str:
    """The value of a Hydra override (``k=v``) or of a flag (``--k v``) inside one command line."""
    tokens = line.split()
    if key.startswith("--"):
        return tokens[tokens.index(key) + 1]

    return next(token for token in tokens if token.startswith(f"{key}=")).split("=", 1)[1]


def test_the_pool_shares_out_the_devices_the_budget_and_the_seeds() -> None:
    """Four workers over three slices, and the extra trials to the workers that start first.

    Devices repeat round-robin, so a pod with fewer slices than workers time-shares them; the seeds
    never repeat, since workers drawing the same start-up points would waste the budget on one
    search done four times.
    """
    result = _launch("--dry-run", *POOL, CATALOGUE, "trainer.max_epochs=3", devices=SLICES)
    assert result.returncode == 0, result.stderr
    lines = _worker_lines(result.stdout)

    assert len(lines) == 4
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
    # Overrides after the catalogue reach every worker, and each worker sweeps the one study.
    assert all("trainer.max_epochs=3" in line for line in lines)
    assert all(_value(line, "--only") == "nconst150" for line in lines)


def test_each_worker_shows_the_command_its_driver_would_run() -> None:
    """The launcher's dry run nests the submitter's, so the assembled python line is visible
    too."""
    result = _launch("--dry-run", *POOL, CATALOGUE, devices=SLICES)
    sweeps = [token for token in result.stdout.split() if token.startswith("hydra.sweep.dir=")]

    assert result.returncode == 0, result.stderr
    assert [sweep.split("_lorentznet_")[1] for sweep in sweeps] == [
        f"nconst150_baseline_t_w{j}" for j in range(4)
    ]


def test_a_dry_run_off_the_pod_stands_in_for_the_devices() -> None:
    """Without the variable there is no pod, and the launch has to be inspectable from a laptop."""
    result = _launch("--dry-run", *POOL, CATALOGUE)

    assert result.returncode == 0, result.stderr
    assert [_value(line, "CUDA_VISIBLE_DEVICES") for line in _worker_lines(result.stdout)] == [
        f"<gpu{j}>" for j in range(4)
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
    ],
)
def test_a_pool_that_cannot_be_shared_out_is_refused(flags: tuple[str, ...]) -> None:
    """Each of these would launch workers that overlap, idle or crash minutes in."""
    assert _launch("--dry-run", *flags, CATALOGUE, devices=SLICES).returncode != 0


def test_a_real_launch_without_devices_is_refused() -> None:
    """A worker with no UUID to pin sees every slice of the pod, and so does the next one."""
    result = _launch("--only", "nconst150", "--trials", "10", "--seed", "7", CATALOGUE)

    assert result.returncode != 0
    assert "NVIDIA_VISIBLE_DEVICES" in result.stderr


def _search_commands(*flags: str) -> list[str]:
    """The commands ``submit_search.sh`` would launch for the LorentzNet catalogue."""
    result = subprocess.run(
        ["bash", SUBMITTER, "--dry-run", *flags, CATALOGUE],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    return [line for line in result.stdout.splitlines() if line.startswith("[")]


def _sweep_dir(command: str) -> str:
    return _value(command, "hydra.sweep.dir")


def _without_sweep_dir(command: str) -> str:
    return " ".join(t for t in command.split() if not t.startswith("hydra.sweep.dir="))


def test_a_tag_moves_the_sweep_directory_and_nothing_else() -> None:
    """Two workers of one study must not merge their multirun outputs, and must not differ in
    anything the search itself sees."""
    plain, tagged = _search_commands(), _search_commands("--tag", "w0")

    assert [_without_sweep_dir(c) for c in tagged] == [_without_sweep_dir(c) for c in plain]
    assert all(_sweep_dir(c).endswith("_w0") for c in tagged)
    assert not any(_sweep_dir(c).endswith("_w0") for c in plain)


def _throwaway_repo(tmp_path: Path) -> Path:
    """A repository root holding the submitter, its library and one catalogue block, nothing else.

    The block names no ``hparams_search``, so the submitter finds no storage URL and creates no
    study database; with ``PYTHON=echo`` the driver it starts writes its own command line and
    exits.
    """
    (tmp_path / ".project-root").touch()
    (tmp_path / "scripts" / "hls4ml").mkdir(parents=True)
    shutil.copytree(ROOT / "scripts" / "cluster", tmp_path / "scripts" / "cluster")
    (tmp_path / "scripts" / "hls4ml" / "runfake_search.sh").write_text(
        "# python3 src/train.py -m experiment=hls4ml/lorentznet_16 \\\n"
        "#     data.method=augmentation data.augmentation_scheme=none\n"
    )

    return tmp_path


def test_a_tagged_worker_leaves_another_workers_live_log_alone(tmp_path: Path) -> None:
    """The submitter rotates the log it is about to write, which is right for a restart and wrong
    for a second worker: on the pods logs/ is one shared volume, so an untagged worker would rotate
    a live log and leave its owner writing to a file nobody reads."""
    repo = _throwaway_repo(tmp_path)
    logdir = repo / "logs" / "searches" / "hls4ml_runfake_search"
    logdir.mkdir(parents=True)
    live = logdir / "1_lorentznet_nconst16_baseline.log"
    live.write_text("a worker of another pod is writing here\n")

    subprocess.run(
        ["bash", SUBMITTER, "--tag", "w1", "scripts/hls4ml/runfake_search.sh"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHON": "echo", "HF_HUB_CACHE": str(tmp_path / "hf")},
    )

    assert live.read_text() == "a worker of another pod is writing here\n"
    assert list(logdir.glob("*.log.*")) == []
    assert "src/train.py" in (logdir / "1_lorentznet_nconst16_baseline_w1.log").read_text()


def test_a_study_is_alive_while_any_of_its_workers_is(tmp_path: Path) -> None:
    """A crashed worker beside a running one leaves the study running, and the copy a restart moved
    aside is history rather than a worker."""
    (tmp_path / "1_lbl.log").write_text("Traceback (most recent call last)\n")
    (tmp_path / "1_lbl_s1_w1.log").write_text("Epoch 3\n")
    (tmp_path / "1_lbl.log.20260101-000000").write_text("Epoch 1\n")

    logs = sweep_status.worker_logs(str(tmp_path), 1, "lbl")

    assert [log.name for log in logs] == ["1_lbl.log", "1_lbl_s1_w1.log"]
    assert sweep_status.classify_workers(logs)[0] == "alive"
    assert sweep_status.classify_workers([])[0] == "no-log"


def test_with_no_worker_alive_the_newest_log_carries_the_verdict(tmp_path: Path) -> None:
    """A worker that exhausted its trials after an older one crashed means a short study, not a
    dead one: the difference is 'short' against 'LAUNCH' in the table."""
    dead = tmp_path / "1_lbl_s1_w0.log"
    dead.write_text("Traceback (most recent call last)\n")
    (tmp_path / "1_lbl_s1_w1.log").write_text("[2026-01-01][HYDRA] Best value: 0.8\n")
    os.utime(dead, (0, 0))

    logs = sweep_status.worker_logs(str(tmp_path), 1, "lbl")

    assert sweep_status.classify_workers(logs)[0] == "FINISHED"
