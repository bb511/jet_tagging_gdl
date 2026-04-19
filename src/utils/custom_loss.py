"""Backward-compatible symmetry loss exports.

This module exists so older model files importing ``src.utils.custom_loss``
continue to work after symmetry helpers moved to ``src.utils.symmetry_loss``.
"""

from src.utils.symmetry_loss import (
    build_symmetric_batch,
    combine_with_symmetry_loss,
    symmetry_distance,
)

__all__ = [
    "build_symmetric_batch",
    "combine_with_symmetry_loss",
    "symmetry_distance",
]
