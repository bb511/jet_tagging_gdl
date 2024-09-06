import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def select_feature_labels(feats: str) -> list[str]:
    """Gets the feature labels for a certain type of selection."""
    all_feature_labels = [
        "$p_x$", "$p_y$", "$p_z$", "$E$", "$E_{rel}$", "$p_T$", "$p_T^{rel}$",
        "$\\eta$", "$\\eta^\\mathrm{rel}$", "$\\eta^\\mathrm{rot}$",
        "$\\phi$", "$\\phi^\\mathrm{rel}$", "$\\phi^\\mathrm{rot}$",
        "$\\Delta_R$", "$\\cos(\\theta)$", "$\\cos(\\theta^\\mathrm{rel})$"
    ]
    
    if feats == "allfeats":
        return all_feature_labels
    else:
        return [feat for feat in feats.split('_') if feat != 'jet' and feat != 'center']

def count_constituents_per_jet(x_data: np.array):
    """Count the number of non-zero constituents for each of the jets in the dataset."""
    return np.sum(np.any(x_data != 0, axis=2), axis=1)

def normalised_data(
    outdir: Path, x_data: np.ndarray, y_data: np.ndarray, train_or_val: str, feats: str
):
    """Plots the data after it has been normalised."""
    print("Plotting the normalised data...")
    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    data_classes = ["Gluon", "Quark", "W", "Z", "Top"]
    feature_labels = select_feature_labels(feats)
    
    for feature in range(x_data.shape[2]):
        plt.figure(figsize=(12, 10))
        
        for class_idx in range(y_data.shape[1]):
            class_mask = np.argmax(y_data, axis=1) == class_idx
            plt.hist(
                x=x_data[class_mask, :, feature].flatten(),
                bins=60,
                alpha=0.5,
                histtype="step",
                linewidth=2.5,
                label=data_classes[class_idx],
                density=True,
                color=colors[class_idx],
            )

        plt.xlabel(feature_labels[feature])
        plt.ylabel("Probability Density")
        plt.yscale("log")
        plt.legend()
        plt.savefig(outdir / f"{feature_labels[feature]}_{train_or_val}_.pdf")
        plt.close()

    print(f"Plots saved to: {outdir}")

def constituent_number(outdir: Path, x_data: np.ndarray, train_or_val: str):
    """Plot the number of constituents in the data that is being normalised per jet."""
    print("Plotting the number of constituents in data...")
    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    constituents_distribution = count_constituents_per_jet(x_data)
    median_const = np.median(constituents_distribution)
    average_const = np.mean(constituents_distribution)

    plt.figure(figsize=(12, 10))
    plt.hist(
        x=constituents_distribution,
        bins=150,
        range=(0, 150),
        alpha=0.4,
        histtype="step",
        linewidth=2.5,
        label=f"Median: {median_const:.1f}\nAverage: {average_const:.3f}",
        color="#648FFF",
    )
    plt.xlabel("Number of Constituents")
    plt.ylabel("Number of Jets")
    plt.legend()
    plt.yscale("log")
    plt.savefig(outdir / f"constituents_plot_{train_or_val}.pdf")
    plt.close()

    print(f"Plots saved to: {outdir}")