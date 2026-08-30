# OmegaConf resolvers the YAML configs call.
"""Custom ``${...}`` resolvers for the Hydra configs.

:func:`register_resolvers` must run before Hydra composes anything, so both entry points call it
immediately after ``rootutils.setup_root``, and the test ``conftest`` calls it at import.
"""

import hashlib

from omegaconf import OmegaConf


def short_hash(value: str, length: int = 12) -> str:
    """Return the first *length* hex characters of the SHA-1 digest of *value*.

    Used to name a run after its own output directory, which is unique per run.
    """
    # str(): under --cfg job no run directory exists and ${hydra:runtime.output_dir} is
    # None, which would otherwise crash the resolved dump of any config using run_name.
    # usedforsecurity=False: this names runs, it does not authenticate anything.
    return hashlib.sha1(str(value).encode("utf-8"), usedforsecurity=False).hexdigest()[:length]


def register_resolvers() -> None:
    """Register every resolver the configs use.

    All are registered with ``replace=True``, so calling this twice (a multirun, or a test module
    re-imported) is harmless.
    """
    # ${len:${data.features}} derives nbr_feats from the feature list, so the two cannot
    # disagree.
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("short_hash", short_hash, replace=True)
