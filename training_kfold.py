import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

import yaml
from sklearn.model_selection import KFold


from import_data import select_data
from import_model import get_model
import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--config_file", type=str, help="Configuration of your model", default="config_equivariant.yml")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--kfold", type=int, default=5)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    batch_accu_sum = 0.0
    num_batches = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, torch.argmax(y, dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y.max(dim=1)[1]) / len(y)
        num_batches += 1
    return running_loss / num_batches, batch_accu_sum / num_batches

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    batch_accu_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, torch.argmax(y, dim=1))
            running_loss += loss.item()
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y.max(dim=1)[1]) / len(y)
            num_batches += 1
    return running_loss / num_batches, batch_accu_sum / num_batches

def k_fold_cross_validation(train_data, net, criterion, config, device, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    results = []
    all_fold_histories = []
    all_fold_predictions = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):
        print(f'FOLD {fold+1}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(train_data, batch_size=config["training_hyperparams"]["batch_size"], sampler=train_subsampler)
        val_loader = DataLoader(train_data, batch_size=config["training_hyperparams"]["batch_size"], sampler=val_subsampler)

        model = get_model(args.name, 32, 3, config["model_hyperparameter"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["training_hyperparams"]["lr"])

        fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(config["training_hyperparams"]["epochs"]):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            print(f'Epoch {epoch+1}/{config["training_hyperparams"]["epochs"]}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc.cpu().item())
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc.cpu().item())

        all_fold_histories.append(fold_history)
        results.append({'fold': fold+1, 'val_accuracy': val_acc.cpu().item()})

        # Collect predictions for ROC and histogram
        model.eval()
        fold_preds = []
        fold_true = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                probs = nn.functional.softmax(outputs, dim=1)
                fold_preds.append(probs.cpu().numpy())
                fold_true.append(y.cpu().numpy())
        
        fold_preds = np.concatenate(fold_preds)
        fold_true = np.concatenate(fold_true)
        all_fold_predictions.append((fold_preds, fold_true))

    avg_accuracy = np.mean([r['val_accuracy'] for r in results])
    std_accuracy = np.std([r['val_accuracy'] for r in results])  
    print(f'Average validation accuracy: {avg_accuracy:.4f} $\\pm$ {std_accuracy:.4f}')

    return results, all_fold_histories, all_fold_predictions

def plot_training_results_with_uncertainty(all_fold_histories, name):
    num_epochs = len(all_fold_histories[0]['train_loss'])
    epochs = range(1, num_epochs + 1)

    metrics = ['loss', 'acc']
    phases = ['train', 'val']

    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(f'Training and Validation Metrics for {name}', fontsize=16)

    for i, metric in enumerate(metrics):
        for j, phase in enumerate(phases):
            key = f'{phase}_{metric}'
            mean_values = np.mean([fold[key] for fold in all_fold_histories], axis=0)
            std_values = np.std([fold[key] for fold in all_fold_histories], axis=0)

            axs[i, j].plot(epochs, mean_values, label=f'Mean {phase.capitalize()} {metric.capitalize()}')
            axs[i, j].fill_between(epochs, mean_values - std_values, mean_values + std_values, alpha=0.2)
            axs[i, j].set_title(f'{phase.capitalize()} {metric.capitalize()}')
            axs[i, j].set_xlabel('Epochs')
            axs[i, j].set_ylabel(metric.capitalize())
            axs[i, j].legend()
            axs[i, j].grid(False)

    plt.tight_layout()
    plt.savefig(f'plot/Training_Results_with_Uncertainty_{name}.png')
    plt.close()


def main(config):
    name = args.name
    data_name = args.data_name
    
    assert name in ["so3_mlp", "mlp", "cg_net", "deepset", "vector_neuron", "so3_canonical", "voxel_net", "scalar", "vector_neuron_big", "vector_neuron_deepset", "jet_cnn", "augerino", "phi_net"], f'the model {name} is not implemented'
    print(f"Training of the model {name} with data {data_name}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    # Setup directories and config
    outdir = util.make_output_directory(f"trained_models_{name}", config["outdir"])
    util.save_config_file(config, outdir)
    config_path = os.path.join(f"trained_models_{name}_{data_name}/trained_models", f"config_{name}_{data_name}.yml")
    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory(f"trained_models_{name}_{data_name}", config["outdir"])

    train_data, test_data = select_data(name, data_name)
    nbr_const, nbr_feat = util.print_data_deets(train_data, "Training")
    
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"]).to(device)

    util.compute_inference_time(model, nbr_const, nbr_feat, name, device, config_path)
    util.check_invariance(model, nbr_const, nbr_feat, name, device, config_path)
    util.check_permutation_invariance(model, nbr_const, nbr_feat, name, device, config_path)

    criterion = nn.CrossEntropyLoss().to(device)

    kfold_results, all_fold_histories, all_fold_predictions = k_fold_cross_validation(train_data, model, criterion, config, device, k=args.kfold)    

    plot_training_results_with_uncertainty(all_fold_histories, f"Training_results{name}_{data_name}")
    util.roc_curves_uncertainty(all_fold_predictions, f"roc_curves_uncertainty_{name}_{data_name}")
    #util.dnn_output_uncertainty("plot",all_fold_predictions, f"{name}_{data_name}_output_histos.pdf")

    # Train final model on all training data
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"])
    train_loader = DataLoader(train_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config["training_hyperparams"]["lr"])
    
    for epoch in range(config["training_hyperparams"]["epochs"]):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{config["training_hyperparams"]["epochs"]}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    # Evaluate on test data
    test_loader = DataLoader(test_data, batch_size=config["training_hyperparams"]["batch_size"])
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    # Save the model
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{name}_{data_name}.pt"))


if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)
    main(config)