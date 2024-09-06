import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

def load_npy_files(directory):
    data = {}
    model_name = None
    for filename in os.listdir(directory):
        if filename.startswith('rotation_accuracy_data_transformer') and filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            parts = filename.split('_')
            model_name = parts[3]
            data_name = '_'.join(parts[4:]).replace('.npy', '')
            print(f"Loading data for {data_name}")
            data[data_name] = np.load(filepath, allow_pickle=True).item()
    return data, model_name

def create_custom_colormap():
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    return LinearSegmentedColormap.from_list("custom", colors, N=len(colors))

def plot_rotation_comparison(data, model_name, output_file):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle(f'Model Accuracy vs. Rotation Angle - {model_name}', y=1.02)

    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation']
    accuracy_keys = ['accuracy_x', 'accuracy_y', 'accuracy_z']

    custom_cmap = create_custom_colormap()
    colors = [custom_cmap(i) for i in np.linspace(0, 1, len(data))]
    
    lines = []
    labels = []

    for ax, title, acc_key in zip(axes, titles, accuracy_keys):
        for (data_name, data_values), color in zip(data.items(), colors):
            angles = data_values['angles']
            accuracies = data_values[acc_key]
            line, = ax.plot(angles, accuracies, marker='o', linestyle='-', linewidth=2.5, markersize=8,
                    color=color)
            
            if acc_key == 'accuracy_x':  # Only add to legend once
                lines.append(line)
                labels.append(data_name)

        ax.set_xlabel('Rotation Angle (degrees)')
        ax.set_ylabel('Accuracy')
        ax.set_title(title, pad=20)
        ax.grid(True, alpha=0.7)
        ax.set_ylim(0.3, 1.0)
        ax.tick_params(axis='both', which='major', labelsize=12)
        #ax.set_facecolor('#f0f0f0')  # Light gray background

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

    # Place legend below the subplots
    fig.legend(lines, labels, loc='lower center', ncol=3, fontsize=12, 
               bbox_to_anchor=(0.5, -0.05), bbox_transform=fig.transFigure)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust bottom to make room for legend
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main(args):
    data, model_name = load_npy_files(args.input_dir)
    
    if not data:
        print("No .npy files found in the specified directory.")
        return
    
    if not model_name:
        print("Could not determine model name from file names.")
        model_name = "Unknown Model"

    plot_rotation_comparison(data, model_name, args.output_file)
    print(f"Enhanced Matplotlib comparison plot saved as {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate enhanced Matplotlib comparison plot from rotation accuracy data files")
    parser.add_argument("--input_dir", type=str, default="./plot", help="Directory containing the .npy files")
    parser.add_argument("--output_file", type=str, default="./plot/enhanced_matplotlib_rotation_accuracy_comparison.png", 
                        help="Output file path for the enhanced comparison plot")
    args = parser.parse_args()
    main(args)