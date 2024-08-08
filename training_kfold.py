import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import yaml
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from import_data import select_data
from import_model import get_model
import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--config_file", type=str, help="Configuration of your model", default="config_equivariant.yml")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--kfold", type=int, default=None, help="Specific fold number to train (0-4)")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

import util
import math


            
def create_balanced_sampler(labels):
    class_sample_count = Counter(labels)
    weights = [1.0 / class_sample_count[label] for label in labels]
    return torch.utils.data.WeightedRandomSampler(weights, len(weights))


def train_model(model, train_loader, criterion, optimizer, config, device):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        y_pred = model(x)
        if config["training_hyperparams"]["loss_inv"] == "so3_eta_phi":
            loss = util.inv_err_rand_eta_phi(model, criterion, x, y_pred, y, 0.01, device)
        else:
            loss = criterion(y_pred, torch.argmax(y, dim=1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += y.size(0)
        correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
    
    return running_loss / len(train_loader), correct / total, model

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                y_pred = model(x)
                loss = criterion(y_pred, torch.argmax(y, dim=1))
            running_loss += loss.item()
            _, predicted = y_pred.max(1)
            total += y.size(0)
            correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
    return running_loss / len(val_loader), correct / total

def train_specific_fold(train_data, criterion, config, device, fold_number, total_folds=5):
    all_labels = []
    for _, y in train_data:
        all_labels.append(torch.argmax(y).item())
    all_labels = np.array(all_labels)
    
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(all_labels)), all_labels))
    
    train_ids, val_ids = folds[fold_number]
    
    print(f'Training FOLD {fold_number}')
    print('--------------------------------')

    train_subset = Subset(train_data, train_ids)
    val_subset = Subset(train_data, val_ids)

    train_loader = DataLoader(train_subset, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["training_hyperparams"]["batch_size"])
    
    # Print class distribution for this fold
    print("Training set class distribution:")
    print(Counter(all_labels[train_ids]))
    print("Validation set class distribution:")
    print(Counter(all_labels[val_ids]))

    x, y = next(iter(train_data))
    nbr_const = x.size(0)
    nbr_feat = x.size(1)
    print(f'Number of constituents: {nbr_const}, Number of features: {nbr_feat}')

    model = get_model(args.name, nbr_const, nbr_feat, config["model_hyperparameter"]).to(device)
    if 'transformer' in args.name.lower():
        util.init_transformer_weights(model)
    else:
        model.apply(util.init_weights)

    
    optimizer = optim.Adam(model.parameters(), lr=config["training_hyperparams"]["lr"])
    if args.name == "vector_neuron":
        optimizer = optim.RMSprop(model.parameters(), lr = 0.0294)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    for epoch in range(config["training_hyperparams"]["epochs"]):
        train_loss, train_acc, model = train_model(model, train_loader, criterion, optimizer, config, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{config["training_hyperparams"]["epochs"]}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        fold_history['train_loss'].append(train_loss)
        fold_history['train_acc'].append(train_acc)
        fold_history['val_loss'].append(val_loss)
        fold_history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'./kfold_model/best_{args.name}_{args.data_name}_fold_{fold_number}.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

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

    return fold_history, (fold_preds, fold_true), val_acc

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
    
    assert name != "transformer", "Use another script for the transformer model"
    print(f"Training of the model {name} with data {data_name}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    # Setup directories and config
    outdir = util.make_output_directory(f"kfold_model/trained_models_{name}", config["outdir"])
    util.save_config_file(config, outdir)
    config_path = os.path.join(f"kfold_model/trained_models_{name}_{data_name}/trained_models", f"config_{name}_{data_name}.yml")
    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory(f"kfold_model/trained_models_{name}_{data_name}", config["outdir"])

    train_data, test_data = select_data(name, data_name)
    nbr_const, nbr_feat = util.print_data_deets(train_data, "Training")
    
    criterion = nn.CrossEntropyLoss().to(device)

    fold_history, fold_predictions, val_accuracy = train_specific_fold(train_data, criterion, config, device, args.kfold)
    
    # Save results for this fold
    np.save(f'fold_history_{args.kfold}.npy', fold_history)
    np.save(f'fold_predictions_{args.kfold}.npy', fold_predictions)
    
    print(f"Fold {args.kfold} completed. Validation accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)
    main(config) 