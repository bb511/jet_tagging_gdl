"""The symmetry groups this study probes, and the names of what is measured under
them."""

from typing import NamedTuple

# Every transform make_transform builds: lorentz boosts along the beam, lorentz_iso
# in a random direction.
GROUPS = ("so2", "so3", "lorentz", "lorentz_iso")

# The domain of data.augmentation_scheme: the groups, plus the unaugmented baseline.
SCHEMES = ("none", *GROUPS)

# The groups the results tables have columns for. lorentz_iso is the full-group check
# LorentzNet  embeds. This latter one is not tabulated.
TABULATED_GROUPS = ("so2", "so3", "lorentz")

# The flavour classes, in the order of the logit columns.
CLASS_NAMES = ("g", "q", "W", "Z", "t")


class WorkingPoint(NamedTuple):
    """A per-class threshold on the clean one-vs-rest score, set by one efficiency."""

    # The suffix of the point's families.
    tag: str
    # "signal" or "background": the efficiency the clean threshold fixes.
    fixed: str
    level: float


# The rejection 1/eps_B at eps_S = 30 % (the top-tagging convention), and the TPR at
# FPR = 1 % and 10 % (JEDI-net, Table 3).
WORKING_POINTS = (
    WorkingPoint("es30", "signal", 0.30),
    WorkingPoint("eb1", "background", 0.01),
    WorkingPoint("eb10", "background", 0.10),
)

# Measured FLIP rates and the accuracy.
FLIP = "flip"
FLIP_RECOVER = "flip_recover"
FLIP_STABLE = "flip_stable"
ACCURACY_TRANSFORMED = "accuracy_transformed"
FLIP_FAMILIES = (FLIP, FLIP_RECOVER, FLIP_STABLE, ACCURACY_TRANSFORMED)


def _measured(point: WorkingPoint) -> str:
    # Fixing the signal efficiency leaves the background rejection to measure, and the
    # other way round.
    return "rejection" if point.fixed == "signal" else "tpr"


def clean_family(point: WorkingPoint) -> str:
    """The groupless family of *point*: its rejection or TPR on the clean scores."""
    return f"{_measured(point)}_{point.tag}"


def retention_family(point: WorkingPoint) -> str:
    """The per-group family of *point*: the share of its clean value that survives the
    transform at the frozen threshold."""
    return f"{_measured(point)}_retention_{point.tag}"


def diagnostic_family(point: WorkingPoint) -> str:
    """The per-group family of *point*: the efficiency the clean threshold fixed, read
    on the transformed scores."""
    fixed = "efficiency" if point.fixed == "signal" else "fpr"
    return f"{fixed}_transformed_{point.tag}"


# Higher is better for the retentions, which enter the deltas. The diagnostics have no
# better direction and never do.
RETENTION_FAMILIES = tuple(retention_family(point) for point in WORKING_POINTS)
DIAGNOSTIC_FAMILIES = tuple(diagnostic_family(point) for point in WORKING_POINTS)

# The per-group families every results table carries, on the judged group and on each.
TABULATED_FAMILIES = FLIP_FAMILIES + RETENTION_FAMILIES + DIAGNOSTIC_FAMILIES

# Paired distances between the clean and the transformed softmax outputs; one of them
# fills the selected and per-group columns of the results table.
TOTAL_VARIATION = "tv"
DISTANCE_FAMILIES = (TOTAL_VARIATION,)

# Groupless: the clean working points and the one-vs-rest AUC.
AUC = "auc"
PERFORMANCE_FAMILIES = tuple(clean_family(point) for point in WORKING_POINTS) + (AUC,)


def key(group: str | None, family: str, class_name: str | None = None) -> str:
    """The metric name the probe logs and the results table reads back.

    A groupless family (*group* None) is a ``performance/`` key. *class_name* names one
    class of a per-class family; without it the key is the macro mean over the classes.
    """
    suffix = f"_{class_name}" if class_name else ""
    if group is None:
        return f"performance/{family}{suffix}"
    return f"invariance/{group}_{family}{suffix}"
