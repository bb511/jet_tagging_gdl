"""The symmetry groups this study probes, and the measured invariance metrics."""

# Every transform make_transform builds: lorentz boosts along the beam, lorentz_iso
# in a random direction.
GROUPS = ("so2", "so3", "lorentz", "lorentz_iso")

# The domain of data.augmentation_scheme: the groups, plus the unaugmented baseline.
SCHEMES = ("none", *GROUPS)

# The groups the results tables have columns for. lorentz_iso is the full-group check
# LorentzNet  embeds. This latter one is not tabulated.
TABULATED_GROUPS = ("so2", "so3", "lorentz")

# Measured FLIP rates and the accuracy.
FLIP = "flip"
FLIP_RECOVER = "flip_recover"
FLIP_STABLE = "flip_stable"
FLIP_DEGRADE = "flip_degrade"
ACCURACY_TRANSFORMED = "accuracy_transformed"
FLIP_FAMILIES = (FLIP, FLIP_RECOVER, FLIP_STABLE, FLIP_DEGRADE, ACCURACY_TRANSFORMED)

# Measures of the whole logit distribution.
WASSERSTEIN = "wasserstein"
DISTANCE_FAMILIES = (WASSERSTEIN,)

FAMILIES = FLIP_FAMILIES + DISTANCE_FAMILIES


def key(group: str, family: str) -> str:
    """The metric name the probe logs and the results table reads back."""
    return f"invariance/{group}_{family}"
