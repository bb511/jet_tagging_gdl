import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from import_data import select_data
from import_model import get_model
import util

def create_rotation_matrix(angle, axis='x'):
    angle_rad = np.radians(angle)
    if axis == 'x':
        return torch.tensor([[1, 0, 0],
                            [0, np.cos(angle_rad), -np.sin(angle_rad)],
                            [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        return torch.tensor([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                            [0, 1, 0],
                            [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        return torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                            [np.sin(angle_rad), np.cos(angle_rad), 0],
                            [0, 0, 1]])

def evaluate_model(model, data_loader, device, rotation_matrix=None):
    print('Evaluating the model')
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x, y_true in data_loader:
            x = x.to(device)
            y_true = y_true.to(device)
            
            # Assuming x shape is (batch_size, num_particles, 5)
            px_py_pz = x[:, :, :3]
            try:
                eta_phi_rel = x[:, :, 3:]
            except:
                eta_phi_rel = None
            
            if rotation_matrix is not None:
                # Apply rotation only to px, py, pz
                batch_size, num_particles, _ = px_py_pz.shape
                rotation_matrix = rotation_matrix.to(device)
                
                # Reshape to (batch_size * num_particles, 3)
                px_py_pz_flat = px_py_pz.reshape(-1, 3)
                
                # Apply rotation
                rotated_px_py_pz_flat = torch.matmul(rotation_matrix, px_py_pz_flat.t()).t()
                
                # Reshape back to (batch_size, num_particles, 3)
                px_py_pz = rotated_px_py_pz_flat.reshape(batch_size, num_particles, 3)
                
            # Recombine rotated px_py_pz with unchanged eta_phi_rel
            if eta_phi_rel is not None:
                x = torch.cat((px_py_pz, eta_phi_rel), dim=-1)
            else:
                x = px_py_pz
            
            y_pred = model(x)
            
            loss = criterion(y_pred, torch.argmax(y_true, dim=1))
            
            total_correct += torch.sum(y_pred.argmax(dim=1) == y_true.argmax(dim=1)).item()
            total_samples += y_true.size(0)
            total_loss += loss.item()
            num_batches += 1

    accuracy = total_correct / total_samples
    avg_loss = total_loss / num_batches

    print(f'Test Accuracy of the model: {accuracy:.2%}')
    print(f'The Test loss of the model: {avg_loss:.4f}')

    return accuracy, avg_loss

def test_individual_rotation(model, sample, device, rotation_matrix):
    model.eval()
    with torch.no_grad():
        original_input = sample[0].to(device)  # Shape: (batch_size, num_particles, 5)
        original_output = model(original_input)  # No need to add batch dimension
        
        # Rotate only px, py, pz
        px_py_pz = original_input[:, :, :3]  # Shape: (batch_size, num_particles, 3)
        rotation_matrix = rotation_matrix.to(device)
        
        # Reshape to (batch_size * num_particles, 3)
        batch_size, num_particles, _ = px_py_pz.shape
        px_py_pz_flat = px_py_pz.reshape(-1, 3)
        
        # Apply rotation
        rotated_px_py_pz_flat = torch.matmul(rotation_matrix, px_py_pz_flat.t()).t()
        
        # Reshape back to (batch_size, num_particles, 3)
        rotated_px_py_pz = rotated_px_py_pz_flat.reshape(batch_size, num_particles, 3)
        
        # Keep eta_rel and phi_rel unchanged
        rotated_input = torch.cat((rotated_px_py_pz, original_input[:, :, 3:]), dim=2)
        
        rotated_output = model(rotated_input)
        
        rotated_input = rotated_input.max(dim=1)[1]
        
        print("Original output:", original_output)
        print("Rotated output:", rotated_output)
        print("Output difference:", torch.abs(original_output - rotated_output).sum().item())

def plot_rotation_accuracy(angles, results_x, results_y, results_z, model_name, data_name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Model Accuracy vs. Rotation Angle - {model_name} ({data_name})', fontsize=16)

    axes = [ax1, ax2, ax3]
    results = [results_x, results_y, results_z]
    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for ax, res, title, color in zip(axes, results, titles, colors):
        ax.plot(angles, res, marker='o', color=color, linestyle='-', linewidth=2, markersize=6)
        ax.set_xlabel('Rotation Angle (degrees)')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.0)  # Changed to 1.0 to show full accuracy range

    plt.tight_layout()
    plt.savefig(f'./plot/rotation_accuracy_{model_name}_{data_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save the data to NPY file
    np.save(f'./plot/rotation_accuracy_data_{model_name}_{data_name}.npy', 
            {'angles': angles, 'accuracy_x': results_x, 'accuracy_y': results_y, 'accuracy_z': results_z})


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing done on {device}")

    path = "/mnt/c/Users/bonvi/Documents/"
    model_name ="sym_loss_sobol_16"
    model_file = path + f"train_models/trained_models_{args.name}_{args.data_name}/trained_models/seed{args.seed}/model_{args.name}_{args.data_name}.pt"
    config_path = path + f"train_models/trained_models_{args.name}_{args.data_name}/trained_models/config.yml"

    config = util.load_config_file(config_path)
    print(f"Testing of the model {args.name} with data {args.data_name} using the config file {config_path}")

    # Import the testing data
    _, test_data = select_data(args.name, args.data_name)
    nbr_const, nbr_feat = util.print_data_deets(test_data, test_data)
    print(util.tcols.OKGREEN, "The Test data has been successfully imported", util.tcols.ENDC)

    # Load the model from file
    model = get_model(args.name, nbr_const, nbr_feat, config["model_hyperparameter"])
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    print(util.tcols.OKGREEN, f"Model {args.name} loaded from {model_file} and moved to {device}", util.tcols.ENDC)

    # Evaluate the model
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, pin_memory=True, num_workers=1)
    evaluate_model(model, test_loader, device)


    # Perform rotation tests
    angles = np.arange(0, 361, 30)
    results_x, results_y, results_z = [], [], []
    for angle in angles:
        for axis, results in zip(['x', 'y', 'z'], [results_x, results_y, results_z]):
            print(f'Rotation Axis: {axis}')
            print(f'Rotation Angle: {angle} degrees')
            print('---------------------------------')
            rotation_matrix = create_rotation_matrix(angle, axis=axis).float()
            accuracy, _ = evaluate_model(model, test_loader, device, rotation_matrix)
            results.append(accuracy)
    
    plot_rotation_accuracy(angles, results_x, results_y, results_z, args.name, args.data_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of the model you want to test")
    parser.add_argument("--data_name", type=str, default='px_py_pz_eta_rel_phi_rel_8')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)