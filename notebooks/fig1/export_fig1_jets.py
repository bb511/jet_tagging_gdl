"""Export candidate jets of the hls4ml test split and their copies under the four groups
at fixed, representative draws, for the panels of Figure 1 of the paper.

Run from the repository root on olgpu with PYTHONPATH=. (the record's loader cache is
read through the datamodule, so the HF snapshot must be in HF_HUB_CACHE). Writes
<out>/fig1_candidates.csv, one row per real constituent and group (GeV, radians, the
transformed columns as the repository's transforms rewrite them), and
<out>/fig1_draws.json with the draws and the selection. --pre-so2-deg and --pre-boost
move the chosen jets into view by a rotation about the beam and a boost along it
before the group copies are made, so that the panels start from the displayed jet.
--iso-axis jet points the isotropic boost of each jet --iso-tilt-deg away from minus
its axis, so that the boost both collimates and moves the jet (a boost perpendicular to
the axis would only move it).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data.components.augmentation import _apply, feature_index, get_norm_scale
from src.data.hls4ml_datamodule import HLS4MLDataModule
from src.symmetry import CLASS_NAMES

OUT_COLUMNS = ["e", "px", "py", "pz", "pt", "eta", "phi", "etarel", "phirel"]
# One representative draw per group, stated in the figure caption; the experiments draw
# uniformly over the full ranges.
DRAWS = {
    "so2": {"angle_rad": 1.0, "axis": [0.0, 0.0, 1.0]},
    "lorentz": {"rapidity": 0.8, "axis": [0.0, 0.0, 1.0]},
    "so3": {"angle_rad": 1.0, "axis": [1.0, 1.0, 1.0]},
    "lorentz_iso": {"rapidity": 0.8, "axis": [1.0, 1.0, 1.0]},
}


def rotation_matrix(axis: list[float], angle: float) -> torch.Tensor:
    """Rodrigues' rotation by *angle* about *axis*, shape (3, 3)."""
    k = torch.tensor(axis, dtype=torch.float64)
    k = k / k.norm()
    cross = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=torch.float64,
    )
    eye = torch.eye(3, dtype=torch.float64)
    return eye + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def boost_matrix(axis: list[float], xi: float) -> torch.Tensor:
    """The boost of rapidity *xi* along *axis* on [E, px, py, pz], as _sample_boosts
    builds it, shape (4, 4)."""
    n = torch.tensor(axis, dtype=torch.float64)
    n = n / n.norm()
    boost = torch.eye(4, dtype=torch.float64)
    boost[0, 0] = math.cosh(xi)
    boost[0, 1:] = -math.sinh(xi) * n
    boost[1:, 0] = -math.sinh(xi) * n
    boost[1:, 1:] += (math.cosh(xi) - 1.0) * torch.outer(n, n)
    return boost


def jet_axes(batch, names, scales) -> torch.Tensor:
    """Unit vectors along each jet's momentum, shape (B, 3)."""
    p = torch.stack(
        [batch[..., feature_index(names, c)] * get_norm_scale(scales, c) for c in ("px", "py", "pz")],
        dim=-1,
    ).sum(dim=1)
    return (p / p.norm(dim=1, keepdim=True)).to(torch.float64)


def iso_axes(jets: torch.Tensor, tilt_deg: float) -> torch.Tensor:
    """Minus each jet's axis, tilted by *tilt_deg* towards the beam: with the sign
    convention of boost_matrix a boost along minus the axis collimates the jet."""
    zhat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    perp = zhat - (jets @ zhat)[:, None] * jets
    perp = perp / perp.norm(dim=1, keepdim=True)
    alpha = math.radians(tilt_deg)
    return -(math.cos(alpha) * jets + math.sin(alpha) * perp)


def group_matrices(batch, names, scales, args) -> dict[str, torch.Tensor]:
    """One matrix per jet and group, shape (B, 3, 3) or (B, 4, 4)."""
    b = batch.shape[0]
    fixed = {
        "so2": rotation_matrix(DRAWS["so2"]["axis"], DRAWS["so2"]["angle_rad"]),
        "lorentz": boost_matrix(DRAWS["lorentz"]["axis"], DRAWS["lorentz"]["rapidity"]),
        "so3": rotation_matrix(DRAWS["so3"]["axis"], DRAWS["so3"]["angle_rad"]),
    }
    xi = DRAWS["lorentz_iso"]["rapidity"]
    if args.iso_axis == "jet":
        axes = iso_axes(jet_axes(batch, names, scales), args.iso_tilt_deg)
        iso = torch.stack([boost_matrix(a.tolist(), xi) for a in axes])
    else:
        fixed["lorentz_iso"] = boost_matrix(DRAWS["lorentz_iso"]["axis"], xi)
    mats = {g: m.unsqueeze(0).expand(b, -1, -1).clone() for g, m in fixed.items()}
    if args.iso_axis == "jet":
        mats["lorentz_iso"] = iso
    return mats


def apply_to_all(batch, mat, names, scales) -> torch.Tensor:
    mats = mat.unsqueeze(0).expand(batch.shape[0], -1, -1).clone()
    return _apply(batch, mats, names, scales)


def axis_angles(x, names, scales) -> tuple[torch.Tensor, torch.Tensor]:
    """The jet axis (eta, phi) in radians, from the leading slot's relative angles."""

    def column(c):
        return x[:, 0, feature_index(names, c)] * get_norm_scale(scales, c)

    eta = column("eta") - column("etarel")
    phi = (column("phi") - column("phirel") + math.pi) % (2 * math.pi) - math.pi
    return eta, phi


def select_candidates(labels, mult, axis, args) -> list[tuple[int, str]]:
    """The requested test indices, or *per_class* indices per requested class inside
    the multiplicity, eta and phi windows."""
    if args.test_indices:
        return [(i, CLASS_NAMES[int(labels[i])]) for i in args.test_indices]
    rng = np.random.default_rng(args.seed)
    eta, phi = axis
    inside = (mult >= args.window[0]) & (mult <= args.window[1])
    if args.eta_window:
        inside &= (eta >= args.eta_window[0]) & (eta <= args.eta_window[1])
    if args.phi_window:
        lo, hi = (math.radians(v) for v in args.phi_window)
        inside &= (phi >= lo) & (phi <= hi)
    picks = []
    for name in args.classes:
        pool = torch.nonzero(inside & (labels == CLASS_NAMES.index(name)))
        pool = pool.squeeze(1).numpy()
        chosen = rng.choice(pool, min(args.per_class, len(pool)), replace=False)
        picks += [(int(i), name) for i in chosen]
    return picks


def rows_of(batch, group, names, scales, picks) -> list[dict]:
    """One row per real constituent of every jet of *batch*, in GeV and radians."""
    columns = {
        c: batch[..., feature_index(names, c)] * get_norm_scale(scales, c)
        for c in OUT_COLUMNS
    }
    rows = []
    for j, (index, name) in enumerate(picks):
        for slot in torch.nonzero(columns["pt"][j] > 0).squeeze(1).tolist():
            rows.append(
                {"candidate": j, "class": name, "test_index": index, "group": group, "slot": slot}
                | {c: float(columns[c][j, slot]) for c in OUT_COLUMNS}
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--classes", nargs="*", default=list(CLASS_NAMES))
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--window", type=int, nargs=2, default=(43, 49))
    parser.add_argument("--eta-window", type=float, nargs=2, default=None)
    parser.add_argument("--phi-window", type=float, nargs=2, default=None, help="deg")
    parser.add_argument("--test-indices", type=int, nargs="*", default=None)
    parser.add_argument("--pre-so2-deg", type=float, default=0.0)
    parser.add_argument("--pre-boost", type=float, default=0.0, help="rapidity")
    parser.add_argument("--iso-axis", choices=["fixed", "jet"], default="fixed")
    parser.add_argument("--iso-tilt-deg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    dm = HLS4MLDataModule(
        revision=args.revision, cache_dir=args.cache_dir, nconstituents=150
    )
    x, y = dm.load_split("test")
    names, scales = dm.feature_names, dm.norm_scales
    mult = (x[..., feature_index(names, "pt")] > 0).sum(dim=1)
    axis = axis_angles(x, names, scales)
    picks = select_candidates(y.argmax(dim=1), mult, axis, args)
    batch = x[[i for i, _ in picks]]
    if args.pre_so2_deg:
        rot = rotation_matrix([0.0, 0.0, 1.0], math.radians(args.pre_so2_deg))
        batch = apply_to_all(batch, rot, names, scales)
    if args.pre_boost:
        batch = apply_to_all(batch, boost_matrix([0.0, 0.0, 1.0], args.pre_boost), names, scales)

    rows = rows_of(batch, "none", names, scales, picks)
    for group, mats in group_matrices(batch, names, scales, args).items():
        rows += rows_of(_apply(batch, mats, names, scales), group, names, scales, picks)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fig1_candidates.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shown = axis_angles(batch, names, scales)
    iso = iso_axes(jet_axes(batch, names, scales), args.iso_tilt_deg) if args.iso_axis == "jet" else None
    selection = {
        "split": "test",
        "nconstituents": 150,
        "classes": args.classes,
        "multiplicity_window": list(args.window),
        "eta_window": args.eta_window,
        "phi_window_deg": args.phi_window,
        "test_indices": args.test_indices,
        "pre_so2_deg": args.pre_so2_deg,
        "pre_boost_rapidity": args.pre_boost,
        "iso_axis_mode": args.iso_axis,
        "iso_tilt_deg": args.iso_tilt_deg if args.iso_axis == "jet" else None,
        "per_class": args.per_class,
        "seed": args.seed,
        "candidates": [
            {
                "candidate": j,
                "class": name,
                "test_index": i,
                "multiplicity": int(mult[i]),
                "axis_eta": float(axis[0][i]),
                "axis_phi_deg": math.degrees(float(axis[1][i])),
                "shown_axis_eta": float(shown[0][j]),
                "shown_axis_phi_deg": math.degrees(float(shown[1][j])),
                "iso_axis": iso[j].tolist() if iso is not None else DRAWS["lorentz_iso"]["axis"],
            }
            for j, (i, name) in enumerate(picks)
        ],
    }
    with open(out / "fig1_draws.json", "w") as handle:
        json.dump({"draws": DRAWS, "selection": selection}, handle, indent=2)
    print(f"wrote {len(rows)} rows for {len(picks)} candidates to {out}")


if __name__ == "__main__":
    main()
