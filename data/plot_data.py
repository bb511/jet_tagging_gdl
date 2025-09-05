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
    outdir: Path | None, x_data: np.ndarray, y_data: np.ndarray, train_or_val: str, feats: str
):
    """Plots the data after it has been normalised."""
    print("Plotting the normalised data...")
    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    # BAD ORDERING
    data_classes = ["Quark", "Gluon", "W", "Z", "Top"]
    feature_labels = select_feature_labels(feats)
    
    if outdir is not None:
        figsize = (3,3)
    else:
        figsize = (12,10)
    
    for feature in range(len(feature_labels)):
        plt.figure(figsize=figsize)
        
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
        if outdir is not None:
            plt.savefig(outdir / f"{feature_labels[feature]}_{train_or_val}_.pdf")
        else:
            plt.show()
            
        plt.close()
    if outdir is not None:
        print(f"Plots saved to: {outdir}")
        
def normalised_data_subplot(
    outdir: Path | None, x_data: np.ndarray, y_data: np.ndarray, train_or_val: str, feats: str, num_cols = 3
):
    """Plots all normalized data features as subplots in a single figure."""
    print("Plotting the normalised data...")

    # Define styles
    plt.rcParams.update({
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
    })

    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    data_classes =  ["Quark", "Gluon", "W", "Z", "Top"]
    feature_labels = select_feature_labels(feats)  # Make sure this function returns valid labels

    num_features = len(feature_labels)  # Number of features to plot
    #num_cols = 3  # Adjust the number of columns per row
    num_rows = int(np.ceil(num_features / num_cols))  # Calculate required rows

    figsize = (num_cols * 6, num_rows * 5)  # Set dynamic figure size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # If we have only one row, axes is a 1D array, we ensure it's always 2D
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    axes = axes.flatten()  # Flatten to loop properly

    for feature in range(num_features):
        ax = axes[feature]

        for class_idx in range(y_data.shape[1]):
            class_mask = np.argmax(y_data, axis=1) == class_idx
            ax.hist(
                x=x_data[class_mask, :, feature].flatten(),
                bins=60,
                alpha=0.5,
                histtype="step",
                linewidth=2.5,
                label=data_classes[class_idx],
                density=True,
                color=colors[class_idx],
            )

        ax.set_xlabel(feature_labels[feature])
        ax.set_ylabel("Probability Density")
        ax.set_yscale("log")
        ax.legend()

    # Hide unused subplots if any
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()

    if outdir is not None:
        save_path = outdir / f"all_features_{train_or_val}.pdf"
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Data histogram plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    
    
def normalised_data_upd2(
    outdir: Path | None,
    x_data_list: list[np.ndarray],
    y_data_list: list[np.ndarray],
    train_or_val: str,
    feats: str,
    dataset_labels: list[str],  # Labels for different datasets
    num_cols=3
):
    """Plots all normalized data features as subplots in a single figure for multiple datasets."""
    print("Plotting the normalised data...")

    # Define styles
    plt.rcParams.update({
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
    })

    # **Updated color scheme: Different colors per dataset**
    base_colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]  # 5 base colors
    dataset_colors = [  # Variations for different datasets
        ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"],  # Normal data
        ["#C9E0FF", "#D9CFF0", "#FFBFD0", "#FFC8A6", "#FFE066"],  # Augmented data (lighter shades)
        ["#2F3EFF", "#5025C0", "#A01F50", "#A04500", "#B07F00"],  # If more datasets exist
    ]
    
    data_classes = ["Gluon", "Quark", "W", "Z", "Top"]
    feature_labels = select_feature_labels(feats)  # Ensure this returns valid labels

    num_features = x_data_list[0].shape[2]  # Assume all datasets have the same features
    num_rows = int(np.ceil(num_features / num_cols))  # Calculate required rows

    figsize = (num_cols * 6, num_rows * 5)  # Set dynamic figure size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # If we have only one row, ensure it's always a 2D array
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    axes = axes.flatten()  # Flatten for easy iteration

    for feature in range(num_features):
        ax = axes[feature]

        for data_idx, (x_data, y_data) in enumerate(zip(x_data_list, y_data_list)):
            dataset_label = dataset_labels[data_idx]  # Get label for this dataset
            dataset_color_set = dataset_colors[data_idx % len(dataset_colors)]  # Pick color set for dataset

            for class_idx in range(y_data.shape[1]):
                class_mask = np.argmax(y_data, axis=1) == class_idx
                ax.hist(
                    x=x_data[class_mask, :, feature].flatten(),
                    bins=60,
                    alpha=0.6,  # Slight transparency to distinguish datasets
                    histtype="step",
                    linewidth=2.5,
                    label=f"{dataset_label} - {data_classes[class_idx]}",  # Correct label format
                    density=True,
                    color=dataset_color_set[class_idx],  # Assign correct color
                )

        ax.set_xlabel(feature_labels[feature])
        ax.set_ylabel("Probability Density")
        ax.set_yscale("log")
        ax.legend()

    # Hide unused subplots if any
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()

    if outdir is not None:
        outdir = Path(outdir) if isinstance(outdir, str) else outdir 
        outdir.mkdir(parents=True, exist_ok=True)
        save_path = outdir / f"all_features_{train_or_val}.pdf"
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    
    
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