import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_npy_files(directory):
    data = {}
    model_name = None
    for filename in os.listdir(directory):
        if filename.startswith('rotation_accuracy_data_') and filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            parts = filename.split('_')
            model_name = parts[3]
            data_name = '_'.join(parts[4:]).replace('.npy', '')  # Join all parts after model name, remove .npy
            data[data_name] = np.load(filepath, allow_pickle=True).item()
    return data, model_name

def plot_rotation_comparison(data, model_name, output_file):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Model Accuracy vs. Rotation Angle - {model_name}', fontsize=16)

    axes = [ax1, ax2, ax3]
    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation']
    accuracy_keys = ['accuracy_x', 'accuracy_y', 'accuracy_z']

    colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))

    for ax, title, acc_key in zip(axes, titles, accuracy_keys):
        for (data_name, data_values), color in zip(data.items(), colors):
            angles = data_values['angles']
            accuracies = data_values[acc_key]
            ax.plot(angles, accuracies, marker='o', linestyle='-', linewidth=2, markersize=6, 
                    label=data_name, color=color)

        ax.set_xlabel('Rotation Angle (degrees)')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0.3, 1.0)
        ax.legend(loc='lower left', fontsize='small')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
    print(f"Comparison plot saved as {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison plot from rotation accuracy data files")
    parser.add_argument("--input_dir", type=str, default="./plot", help="Directory containing the .npy files")
    parser.add_argument("--output_file", type=str, default="./plot/rotation_accuracy_comparison.png", 
                        help="Output file path for the comparison plot")
    args = parser.parse_args()
    main(args)