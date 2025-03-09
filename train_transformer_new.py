import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--config_file", type=str, help ="What are the configuration of your model", default= "config_equivariant.yml")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--loss", type=str, default='default', choices=['default', 'so3', 'lorentz', 'jet_center', 'so3_eta_phi'], help = "Type of loss function to use")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import random

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

import yaml
import math

from import_data import select_data
from import_model import get_model
import data.standardization as standardization

import custom_loss

import util
import plot_train

def main(config):
    name = args.name
    assert name == "transformer", "This training script is only for the transformer model"
    data_name = args.data_name
    loss_name = args.loss

    # Set up the file name
    if loss_name == 'default':
        file_name = f"{name}_{data_name}"
    else:
        file_name = f"{name}_{data_name}_{loss_name}"
    print(f"Training of the model {name} with data {data_name} and loss {loss_name}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    plotdir = util.make_output_directory("plot", file_name)
    outdir = util.make_output_directory(f"train_models/trained_models_{name}_{data_name}", config["outdir"])
    
    # Save the config file to the main dir.
    util.save_config_file(config, outdir)
    config_path = os.path.join(outdir, f"config_{name}_{data_name}_{loss_name}.yml")
    print(f'Config file saved in {config_path}')

    # Create subdir for each seed.
    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory(f"train_models/{file_name}", config["outdir"])
    print('Subfile for each seeds are created')

    # Import the training and the validation data, sampled and normalised.
    print(util.tcols.OKBLUE + "Importing data..." + util.tcols.ENDC)
    training_data, _ = select_data(name, data_name)
    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size
    
    # Load normalization coefficients
    coef_norm = standardization.load_normalization_coeff(data_name)
    
    # Use random_split to create the split
    train_dataset, val_dataset = torch.utils.data.random_split(
        training_data, 
        [train_size, val_size],
    )
    print(util.tcols.OKGREEN + "Data imported." + util.tcols.ENDC)
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")


    print(f"Training data shape: {train_dataset[:][0].shape}")
    nbr_const = train_dataset[:][0].shape[1]
    nbr_feat = train_dataset[:][0].shape[2]
    print(f"Training labels shape: {train_dataset[:][1].shape}")
    print(f"Validation data shape: {val_dataset[:][0].shape}")
    print(f"Validation labels shape: {val_dataset[:][1].shape}")
    
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory = True, num_workers=1)
    valid_load = torch.utils.data.DataLoader(val_dataset, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory = True, num_workers=1)

    # Import and train the model.
    print("")
    print(util.tcols.OKBLUE + "Importing model..." + util.tcols.ENDC)
    model = get_model(name, nbr_const, nbr_feat, coef_norm, config["model_hyperparameter"]).to(device) # TODO add the normalzation inside the model
    model.apply(util.init_weights)
    
    print(util.tcols.OKCYAN + 'Model summary: ' + util.tcols.ENDC, '\n', model)

    # Check different properties of the model.
    x, _ = next(iter(train_load))
    
    print("")
    print(util.tcols.OKBLUE + "Model specification in terms of inference time, FLOPs and number of parameters" + util.tcols.ENDC)
    util.compute_inference_time(model, x, device)
    util.compute_flops(model, x, device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print("")
    plot_train.weights_plot(model, plotdir, file_name + '_init')
    print("")
    print(util.tcols.OKBLUE + "Checking the invariance properties of the model..." + util.tcols.ENDC)
    if "px_py_pz_E" not in data_name:
        util.check_invariance(model, x, nbr_rot=32, device=device)
    else:
        util.check_invariance_Lorentz(model, x, nbr_rot=32, device=device)
    util.check_permutation_invariance(model, x, nbr_perm=32, device=device)
    print("")

    
    train_model(model, train_load, valid_load, device, name, data_name, loss_name, plotdir, **config["training_hyperparams"], warmup_steps=1000)
    print('The model has been successfully trained')
    
    # Saved model and plot the loss and accuracy evolution throught the epochs.
    model_file = os.path.join(outdir, f"{file_name}.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")


    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """Custom learning rate scheduler for transformers"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(net, train_data, valid_data, device, name, data_name, loss_inv, plotdir, epochs, lr, batch_size, warmup_steps, l = 0.001):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = len(train_data) * epochs
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    print(f'Beginning of training of model {name}')
    all_train_loss, all_train_accu, all_valid_loss, all_valid_accu, all_inv_err = [], [], [], [], []
    epochs_no_improve = 0
    epochs_es_limit = 50
    best_accu = 0
    
    for epoch in range(epochs):
        batch_loss_sum, batch_accu_sum, totnum_batches = 0, 0, 0
        net.train()
        print(f'Beginning of Epoch {epoch + 1}/{epochs}')
        
        for x, y_true in train_data:
            x, y_true = x.to(device), y_true.to(device)
            print(f'The shape of x is {x.shape}')
            
            y_pred = net(x)
            
            if loss_inv == 'so3':
                loss = custom_loss.inv_err_randn(net, criterion, x, y_pred, y_true, 0.001, device)
            elif loss_inv == 'lorentz':
                loss = custom_loss.inv_err_lorentz(net, criterion, x, y_pred, y_true, 0.001, device)
            elif loss_inv == 'jet_center':
                loss = custom_loss.inv_err_px_py_pz_jet_center(net, criterion, x, y_pred, y_true, 0.001, device)
            elif loss_inv == 'phi':
                loss = custom_loss.inv_err_phi(net, criterion, x, y_pred, y_true, 0.001, device)
            elif loss_inv == 'so3_eta_phi':
                loss = custom_loss.inv_err_rand_eta_phi(net, criterion, x, y_pred, y_true, l, device)
            else:
                loss = criterion(y_pred, torch.argmax(y_true, dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            batch_loss_sum += loss.detach()
            totnum_batches += 1
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
            
        
        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches
        if 'px_py_pz' in data_name:
            inv_err = util.check_inv(net, x, device)
        else:
            inv_err = 100
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        all_inv_err.append(inv_err.item())
        
        
        print(f'[{epoch + 1}] Trained loss: {train_loss:.5f}')
        print(f'[{epoch + 1}] Trained accuracy: {train_accu:.5f}')
        print(f'[{epoch + 1}] Invariance relative error: {inv_err:.5f}')
        print(f'[{epoch + 1}] Learning rate: {scheduler.get_last_lr()[0]}')
        
        batch_loss_sum, batch_accu_sum, totnum_batches = 0, 0, 0
        
        with torch.no_grad():
            net.eval()
            for x, y in valid_data:
                x, y = x.to(device), y.to(device)
                
                y_pred = net(x)
                y_true = y
                
                loss = criterion(y_pred, torch.argmax(y_true, dim=1).to(torch.int64))
                
                batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
                batch_loss_sum += loss.detach()
                totnum_batches += 1
            
            valid_loss = batch_loss_sum / totnum_batches
            valid_accu = batch_accu_sum / totnum_batches
            print(f"[{epoch + 1}] validation loss: {valid_loss:.5f}")
            print(f"[{epoch + 1}] test accuracy: {valid_accu:.5f}")
            all_valid_loss.append(valid_loss.item())
            all_valid_accu.append(valid_accu.item())
        
        if round(valid_accu.cpu().item(), 3) <= round(best_accu, 3):
            epochs_no_improve += 1
        else:
            best_accu = valid_accu.cpu().item()
            epochs_no_improve = 0
        scheduler.step(best_accu)
        if epochs_no_improve >= epochs_es_limit:
            break
    
    plot_train.plot_loss_accu(all_train_loss, all_train_accu, all_valid_loss, all_valid_accu, all_inv_err, name, data_name, loss_inv, plotdir)


if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    main(config)