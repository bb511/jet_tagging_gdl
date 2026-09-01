"""Custom ``${...}`` resolvers for the Hydra configs.

:func:`register_resolvers` must run before Hydra composes anything, hence the call at the top of
both entry points and in the test ``conftest``.
"""

import hashlib

from omegaconf import OmegaConf


def short_hash(value: str, length: int = 12) -> str:
    """Return the first *length* hex characters of the SHA-1 digest of *value*.

    Used to name a run after its own output directory, which is unique per run.
    """
    # str(): under --cfg job ${hydra:runtime.output_dir} is None, which would crash the dump.
    # usedforsecurity=False: this names runs, it does not authenticate anything.
    return hashlib.sha1(str(value).encode("utf-8"), usedforsecurity=False).hexdigest()[:length]


def register_resolvers() -> None:
    """Register every resolver; ``replace=True`` makes a second call harmless."""
    # ${len:${data.features}} derives nbr_feats from the feature list, so the two cannot disagree.
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("short_hash", short_hash, replace=True)
