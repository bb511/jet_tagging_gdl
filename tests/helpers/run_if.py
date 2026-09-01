"""Adapted from:

https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/runif.py
"""

import sys
from importlib.metadata import version as _distribution_version
from typing import Any

import pytest
import torch
from packaging.version import Version
from pytest import MarkDecorator

from tests.helpers.package_available import _IS_WINDOWS


class RunIf:
    """Conditional skipping of tests, fully compatible with `@pytest.mark`.

    ```python
        @RunIf(min_gpus=1)
        def test_wrapper():
            ...
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        min_torch: str | None = None,
        max_torch: str | None = None,
        min_python: str | None = None,
        skip_windows: bool = False,
        **kwargs: dict[Any, Any],
    ) -> MarkDecorator:
        """Create the `skipif` mark; *kwargs* are passed to `pytest.mark.skipif` untouched."""
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            conditions.append(Version(_distribution_version("torch")) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            conditions.append(Version(_distribution_version("torch")) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if skip_windows:
            conditions.append(_IS_WINDOWS)
            reasons.append("does not run on Windows")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
