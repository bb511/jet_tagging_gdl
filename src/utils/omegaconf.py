"""Custom ``${...}`` resolvers for the Hydra configs."""

import hashlib

from omegaconf import OmegaConf


def short_hash(value: str, length: int = 12) -> str:
    """Return the first *length* hex characters of the SHA-1 digest of *value*.

    Used to name a run after its own output directory, which is unique per run.
    """
    # str(): under --cfg job ${hydra:runtime.output_dir} is None, which would crash the dump.
    # usedforsecurity=False: this names runs, it does not authenticate anything.
    return hashlib.sha1(str(value).encode("utf-8"), usedforsecurity=False).hexdigest()[
        :length
    ]


def fold_label(fold: int | None, default: str) -> str:
    """Return ``fold_<i>`` for fold *fold*, counted from one, or *default* for main.

    Names a run after the one ``fold`` knob of ``train.yaml``, in the manifest and in
    MLflow.
    """
    return default if fold is None else f"fold_{int(fold)}"


def register_resolvers() -> None:
    """Register every resolver; ``replace=True`` makes a second call harmless."""
    # ${len:${data.features}} derives nbr_feats from the feature list.
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("short_hash", short_hash, replace=True)
    OmegaConf.register_new_resolver("fold_label", fold_label, replace=True)
