from omegaconf import OmegaConf

from src.utils.omegaconf import fold_label, register_resolvers, short_hash

RUN_DIR = "/logs/train/runs/2026-08-29_12-00-00"


def test_the_resolvers_count_the_features_and_name_the_run() -> None:
    # conftest registered them once already; a multirun calls this a second time too.
    register_resolvers()
    cfg = OmegaConf.create(
        {
            "data": {"features": ["px", "py", "pz", "e"]},
            "nbr_feats": "${len:${data.features}}",
        }
    )

    assert cfg.nbr_feats == 4
    cfg.data.features = ["px", "py", "pz"]
    assert cfg.nbr_feats == 3

    assert len(short_hash(RUN_DIR, 12)) == 12
    assert len(short_hash(RUN_DIR, 8)) == 8
    # One run keeps its name on a resume, and two runs are told apart.
    assert short_hash(RUN_DIR) == short_hash(RUN_DIR)
    assert short_hash(RUN_DIR) != short_hash("/logs/train/runs/2026-08-29_12-00-01")

    # One knob, fold, names the run: the main run keeps the default, a fold its number.
    cfg = OmegaConf.create({"fold": None, "label": "${fold_label:${fold},main}"})
    assert cfg.label == "main"
    cfg.fold = 3
    assert cfg.label == "fold_3"
    assert fold_label(None, "abc") == "abc"
