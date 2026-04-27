from typing import Tuple

import numpy as np 



def permute_constituents(
    x_data: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomly permute constituent order within each jet."""
    n_jets, n_const = x_data.shape[:2]
    # Generate all permutations at once via argsort of random values
    perms = np.argsort(rng.random((n_jets, n_const)), axis=1)
    return x_data[np.arange(n_jets)[:, None], perms]

def apply_random_permutation(
    x_data: np.ndarray,
    y_data: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Permute constituent order for every jet and return the shuffled dataset."""
    return permute_constituents(x_data, rng), y_data


