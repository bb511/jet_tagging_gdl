import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
import yaml

from import_data import select_data
from import_model import get_model

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
            
            if rotation_matrix is not None:
                x = torch.bmm(x, rotation_matrix.repeat(x.size(0), 1, 1).to(device))
            
            y_pred = model(x)
            
            loss = criterion(y_pred, y_true)
            
            total_correct += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]).item()
            total_samples += y_true.size(0)
            total_loss += loss.item()
            num_batches += 1

    accuracy = total_correct / total_samples
    avg_loss = total_loss / num_batches

    print(f'Test Accuracy of the model: {accuracy:.2%}')
    print(f'The Test loss of the model: {avg_loss:.4f}')

    return accuracy, avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'deepset'
    data_name = 'pt_eta_rel_phi_rel_8'

    data, _ = select_data(model_name, data_name)

    x, y = next(iter(data))
    nbr_const = x.size(0)
    nbr_feat = x.size(1)
    print(f'Number of constituents: {nbr_const}')
    print(f'Number of features: {nbr_feat}')

    config_file = "config_equivariant.yml"

    with open(config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    model = get_model(model_name, nbr_const, nbr_feat, config["model_hyperparameter"])
    
    path = '/mnt/c/Users/bonvi/Documents/master_thesis/master_thesis_code/trained_models_deepset_pt_eta_rel_phi_rel_8/trained_models_pt_eta_phi_jet_center_8const/seed42/model_deepset_pt_eta_rel_phi_rel_8.pt'

    model.load_state_dict(torch.load(path))
    model = model.to(device)
    
    data = DataLoader(data, batch_size=256, shuffle=False)
            
    angles = np.arange(0, 361, 30)
    rotation_matrices_x = {angle: create_rotation_matrix(angle, axis='x') for angle in angles}
    rotation_matrices_y = {angle: create_rotation_matrix(angle, axis='y') for angle in angles}
    rotation_matrices_z = {angle: create_rotation_matrix(angle, axis='z') for angle in angles}
    

    results_x = []
    results_y = []
    results_z = []
    for angle in angles:
        rotation_matrix_x = rotation_matrices_x[angle].float().to(device)
        rotation_matrix_y = rotation_matrices_y[angle].float().to(device)
        rotation_matrix_z = rotation_matrices_z[angle].float().to(device) 
        accuracy, _ = evaluate_model(model, data, device, rotation_matrix_x)
        results_x.append(accuracy)
        accuracy, _ = evaluate_model(model, data, device, rotation_matrix_y)
        results_y.append(accuracy)
        accuracy, _ = evaluate_model(model, data, device, rotation_matrix_z)
        results_z.append(accuracy)
            
    plt.figure(figsize=(12, 8))
    plt.plot(angles, results_x, marker='o', label=model_name)

    plt.xlabel('Rotation Angle in the x axis (degrees)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Rotation Angle')
    plt.legend()
    plt.savefig(f'./plot/rotation_accuracy_x_{model_name}_{data_name}.png')

    plt.figure(figsize=(12, 8))
    plt.plot(angles, results_y, marker='o', label=model_name)

    plt.xlabel('Rotation Angle in the y axis (degrees)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Rotation Angle')
    plt.legend()
    plt.savefig(f'./plot/rotation_accuracy_y_{model_name}_{data_name}')

    plt.figure(figsize=(12, 8))
    plt.plot(angles, results_z, marker='o', label=model_name)

    plt.xlabel('Rotation Angle in the z axis (degrees)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Rotation Angle')
    plt.legend()
    plt.savefig(f'./plot/rotation_accuracy_z_{model_name}_{data_name}')
    
if __name__ == "__main__":
    main()
