#!/usr/bin/env python3
"""Replot constituent-feature overlap histograms without running training.

Usage
-----
    python scripts/replot_overlap.py                          # lorentz, nconst=32, class 0
    python scripts/replot_overlap.py --scheme so2 --nconst 16
    python scripts/replot_overlap.py --scheme lorentz --nconst 8 --class_idx 1 --n_jets 10000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT    = Path(__file__).resolve().parents[2]
DATASETS_DIR = "/cluster/scratch/agogia/datasets"
SAVE_DIR     = str(REPO_ROOT / "logs" / "plots")

# Math-formatted axis labels for known features, keyed without the "j1_" prefix.
FEATURE_LABELS = {
    "px": r"$p_x$",
    "py": r"$p_y$",
    "pz": r"$p_z$",
    "e": r"$E$",
    "erel": r"$E_{\mathrm{rel}}$",
    "pt": r"$p_T$",
    "ptrel": r"$p_{T,\mathrm{rel}}$",
    "eta": r"$\eta$",
    "etarel": r"$\eta_{\mathrm{rel}}$",
    "etarot": r"$\eta_{\mathrm{rot}}$",
    "phi": r"$\phi$",
    "phirel": r"$\phi_{\mathrm{rel}}$",
    "phirot": r"$\phi_{\mathrm{rot}}$",
    "deltar": r"$\Delta R$",
    "costheta": r"$\cos\theta$",
    "costhetarel": r"$\cos\theta_{\mathrm{rel}}$",
}

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def _math_label(feat_name: str) -> str:
    """Return a math-formatted axis label for a feature name like 'j1_pt'."""
    key = feat_name[len("j1_"):] if feat_name.startswith("j1_") else feat_name
    return FEATURE_LABELS.get(key.lower(), feat_name)

sys.path.insert(0, str(REPO_ROOT))

from src.utils.augmentation import (
    apply_so2_batched,
    apply_so3_batched,
    apply_lorentz_batched,
    create_lorentz_boosts,
    create_so2_rot_matrices,
    create_so3_rot_matrices,
    feature_index,
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _pt_real_mask(x: torch.Tensor, feat_names: list) -> torch.Tensor:
    """Boolean mask (N*C,) that is True for real (non-padded) constituents."""
    pt_feat = next((f for f in ("j1_pt", "pt") if f in feat_names), None)
    if pt_feat is not None:
        pt_idx = feat_names.index(pt_feat)
        return (x[:, :, pt_idx] > 0).reshape(-1)
    return torch.ones(x.shape[0] * x.shape[1], dtype=torch.bool)


def _pct_outside_bounds(orig_vals: np.ndarray, aug_vals: np.ndarray) -> float:
    """% of aug_vals falling outside [min(orig_vals), max(orig_vals)]."""
    lo, hi = orig_vals.min(), orig_vals.max()
    outside = np.count_nonzero((aug_vals < lo) | (aug_vals > hi))
    return 100.0 * outside / len(aug_vals)


def _plot(
    x_orig: torch.Tensor,
    augmented: dict[str, torch.Tensor],   # label -> tensor
    feat_names: list[str],
    real_mask_orig: torch.Tensor,
    output_path: Path,
    bins: int = 60,
    num_cols: int = 3,
    plot_features: list[str] | None = None,
) -> list[dict]:
    colours = ["red", "blue", "green", "orange"]

    if plot_features is not None:
        f_indices = [feature_index(feat_names, f) for f in plot_features]
        missing = [f for f, i in zip(plot_features, f_indices) if i is None]
        if missing:
            print(f"[warn] Features not found, skipping: {missing}")
        f_indices = [i for i in f_indices if i is not None]
    else:
        f_indices = list(range(x_orig.shape[2]))

    num_features = len(f_indices)
    num_rows = math.ceil(num_features / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
    axes = np.atleast_1d(axes).flatten()

    bounds_summary = []
    for ax_idx, f_idx in enumerate(f_indices):
        ax = axes[ax_idx]
        orig_vals = x_orig[:, :, f_idx].reshape(-1)[real_mask_orig].numpy()
        ax.hist(orig_vals, bins=bins, density=True, histtype="step",
                linewidth=2.5, label="Original", color="black")

        for (label, x_aug), colour in zip(augmented.items(), colours):
            real_mask_aug = _pt_real_mask(x_aug, feat_names)
            aug_vals = x_aug[:, :, f_idx].reshape(-1)[real_mask_aug].numpy()
            pct_outside = _pct_outside_bounds(orig_vals, aug_vals)
            bounds_summary.append({
                "feature": feat_names[f_idx],
                "variant": label,
                "pct_outside_orig_bounds": pct_outside,
            })
            ax.hist(aug_vals, bins=bins, density=True, histtype="step",
                    linewidth=2.0, label=label, color=colour)

        ax.set_xlabel(_math_label(feat_names[f_idx]))
        ax.set_ylabel("Probability Density")
        ax.set_yscale("log")
        ax.legend(fontsize=13)
        ax.tick_params(labelsize=14)

    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path.resolve()}")
    return bounds_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheme",    default="lorentz",
                        choices=["lorentz", "so2", "so3"])
    parser.add_argument("--nconst",   type=int, default=32)
    parser.add_argument("--class_idx",type=int, default=0)
    parser.add_argument("--n_jets",   type=int, default=5120,
                        help="Number of jets to sample (before class filter).")
    parser.add_argument("--eta_max",  type=float, default=2.0,
                        help="Max rapidity for Lorentz boosts.")
    parser.add_argument("--bins",     type=int, default=60)
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument(
        "--features", nargs="+",
        default=None,
        help="Subset of features to plot (default: all available features).",
    )
    args = parser.parse_args()

    processed_dir = (
        Path(DATASETS_DIR) / "processed" / "none" / f"nconst_{args.nconst}"
    )

    print(f"Loading data from {processed_dir} …")
    # mmap instead of a full load — only the rows we sample below ever get read from disk.
    x_full = np.load(processed_dir / "x_train.npy", allow_pickle=False, mmap_mode="r")
    y_full = np.load(processed_dir / "y_train.npy", allow_pickle=False, mmap_mode="r")

    npz = np.load(processed_dir / "norm_params.npz", allow_pickle=False)
    feat_names  = list(npz["feature_names"])
    norm_scales = {n: float(s) for n, s in zip(npz["feature_names"], npz["scales"])}

    # Sample jets
    rng = np.random.default_rng()
    n_samples = min(args.n_jets, len(x_full))
    idx = rng.choice(len(x_full), n_samples, replace=False)
    x_sample = torch.from_numpy(x_full[idx]).float()
    y_sample  = y_full[idx]

    # Filter to requested class
    class_ids = np.argmax(y_sample, axis=1) if y_sample.ndim > 1 else y_sample.astype(int)
    mask = class_ids == args.class_idx
    x_class = x_sample[mask]
    print(f"Class {args.class_idx}: {int(mask.sum())} jets after filtering.")

    if x_class.shape[0] == 0:
        sys.exit(f"No jets found for class {args.class_idx}.")

    real_mask_orig = _pt_real_mask(x_class, feat_names)
    nconst_tag = f"nconst_{args.nconst}"
    save_dir   = Path(args.save_dir)

    # ---- Lorentz: plot both with and without SO(2) phi correction ----
    if args.scheme == "lorentz":
        boosts = create_lorentz_boosts(
            x_class.shape[0], args.eta_max, rng, device=x_class.device, dtype=x_class.dtype
        )

        x_no_so2 = apply_lorentz_batched(x_class, boosts, feat_names, norm_scales)

        bounds_summary = _plot(
            x_orig=x_class,
            augmented={"Augmented": x_no_so2},
            feat_names=feat_names,
            real_mask_orig=real_mask_orig,
            output_path=save_dir / f"overlap_lorentz_comparison_{nconst_tag}_class_{args.class_idx}.png",
            bins=args.bins,
            plot_features=args.features,
        )

    # ---- SO(2) ----
    elif args.scheme == "so2":
        mats = create_so2_rot_matrices(x_class.shape[0], rng, device=x_class.device, dtype=x_class.dtype)
        x_aug = apply_so2_batched(x_class, mats, feat_names, norm_scales)
        bounds_summary = _plot(
            x_orig=x_class,
            augmented={"SO(2) augmented": x_aug},
            feat_names=feat_names,
            real_mask_orig=real_mask_orig,
            output_path=save_dir / f"overlap_so2_{nconst_tag}_class_{args.class_idx}.png",
            bins=args.bins,
            plot_features=args.features,
        )

    # ---- SO(3) ----
    elif args.scheme == "so3":
        mats = create_so3_rot_matrices(x_class.shape[0], rng, device=x_class.device, dtype=x_class.dtype)
        x_aug = apply_so3_batched(x_class, mats, feat_names, norm_scales)
        bounds_summary = _plot(
            x_orig=x_class,
            augmented={"SO(3) augmented": x_aug},
            feat_names=feat_names,
            real_mask_orig=real_mask_orig,
            output_path=save_dir / f"overlap_so3_{nconst_tag}_class_{args.class_idx}.png",
            bins=args.bins,
            plot_features=args.features,
        )

    print(f"\n{'Feature':<12} {'Variant':<20} {'% outside orig. bounds':>22}")
    print("-" * 56)
    for row in bounds_summary:
        print(f"{row['feature']:<12} {row['variant']:<20} {row['pct_outside_orig_bounds']:>21.2f}%")

    summary_csv = save_dir / f"overlap_bounds_{args.scheme}_{nconst_tag}_class_{args.class_idx}.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w") as f:
        f.write("feature,variant,pct_outside_orig_bounds\n")
        for row in bounds_summary:
            
            f.write(f"{row['feature']},{row['variant']},{row['pct_outside_orig_bounds']:.4f}\n")
    print(f"\nSaved bounds summary → {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
