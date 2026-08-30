"""The custom ``${...}`` resolvers, which several configs cannot be composed without."""

from omegaconf import OmegaConf

from src.utils.omegaconf import register_resolvers, short_hash


def test_len_derives_the_feature_count_from_the_feature_list() -> None:
    """Every algorithm config sets nbr_feats this way, so the two cannot disagree."""
    cfg = OmegaConf.create(
        {"data": {"features": ["px", "py", "pz", "e"]}, "nbr_feats": "${len:${data.features}}"}
    )

    assert cfg.nbr_feats == 4

    cfg.data.features = ["px", "py", "pz"]
    assert cfg.nbr_feats == 3


def test_short_hash_returns_the_requested_number_of_characters() -> None:
    """It names an MLflow run, where a full SHA-1 digest is unreadable."""
    assert len(short_hash("/logs/train/runs/2026-08-29_12-00-00", 12)) == 12
    assert len(short_hash("/logs/train/runs/2026-08-29_12-00-00", 8)) == 8


def test_short_hash_is_deterministic_and_separates_two_run_directories() -> None:
    """Two runs must be told apart by their names, and one run named the same on a resume."""
    first = short_hash("/logs/train/runs/2026-08-29_12-00-00")
    again = short_hash("/logs/train/runs/2026-08-29_12-00-00")
    other = short_hash("/logs/train/runs/2026-08-29_12-00-01")

    assert first == again
    assert first != other


def test_registering_the_resolvers_twice_is_harmless() -> None:
    """A multirun and a re-imported test module both call it a second time."""
    register_resolvers()
    register_resolvers()

    assert OmegaConf.create({"n": "${len:[1, 2, 3]}"}).n == 3
