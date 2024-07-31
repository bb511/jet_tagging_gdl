import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--config_file", type=str, help ="What are the configuration of your model", default= "config_equivariant.yml")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--seed", type=int, default=42)
#parser.add_argument("--device", type=str, default='cuda:0')
args = parser.parse_args()

import random
import math

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

import torch_geometric

import yaml

import deepspeed

from import_data import select_data
from import_model import get_model

import util
import pickle

def main(config):
    name = args.name
    assert name == "transformer", "This training script is only for the transformer model"
    data_name = args.data_name
    print(f"Training of the model {name} with data {data_name}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    outdir = util.make_output_directory(f"train_models/trained_models_{name}_{data_name}", config["outdir"])
    # Save the config file to the main dir.
    util.save_config_file(config, outdir)
    config_path = os.path.join(outdir, f"config_{name}_{data_name}.yml")
    print('Config file saved')

    # Create subdir for each seed.
    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory(f"train_models/trained_models_{name}_{data_name}", config["outdir"])
    print('Subfile for each seeds are created')

    # Import the training and the validation data, sampled and normalised.
    train_data, valid_data = select_data(name, data_name)
    nbr_const, nbr_feat = util.print_data_deets(train_data, "Training")
    util.print_data_deets(valid_data, "Validation")
    print("The training data and the validation data have been successfully imported")
    train_load = torch.utils.data.DataLoader(train_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory=True, num_workers=1)
    valid_load = torch.utils.data.DataLoader(valid_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory=True, num_workers=1)
    x, y = next(iter(train_load))
    # Import and train the model.
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"]).to(device)
    print('Model summary: ', model)
    util.compute_inference_time(model, x, name, device, config_path)
    util.weights_plot(model, name + data_name + '_init')
    util.check_permutation_invariance(model, x, name, device, config_path)

    train_model(model, train_load, valid_load, device, name, data_name, **config["training_hyperparams"], warmup_steps=1000)
    print('The model has been successfully trained')
    
    # Saved model and plot the loss and accuracy evolution throught the epochs.
    model_file = os.path.join(outdir, f"model_{name}_{data_name}.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")


    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(net, train_data, valid_data, device, name, data_name, epochs, lr, batch_size, loss_inv, warmup_steps):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)

    
    total_steps = len(train_data) * epochs
    
    # Create scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode="max",
                                                            patience= 80,
                                                            factor=0.1,
                                                            threshold=1e-4,
                                                            verbose=True,
                                                            )
    
    print(f'Beginning of training of model {name}')
    all_train_loss, all_train_accu, all_valid_loss, all_valid_accu, all_inv_err = [], [], [], [], []
    epochs_no_improve = 0
    epochs_es_limit = 50
    best_accu = 0
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    print("")
    
    prof = deepspeed.profiling.flops_profiler.FlopsProfiler(net)
    
    for epoch in range(epochs):
        batch_loss_sum, batch_accu_sum, totnum_batches = 0, 0, 0
        net.train()
        print(f'Beginning of Epoch {epoch + 1}/{epochs}')
        
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            y_true = y
            
            if epoch == 0 and totnum_batches == 0:
                prof.start_profile()
                y_pred = net(x)
                prof.stop_profile()
                print(util.tcols.OKGREEN + "Model flops: " + util.tcols.ENDC, end="")
                print(f"{prof.get_total_flops()/ x.shape[0]}")
                prof.end_profile()
                print("")
            else:
                y_pred = net(x)
            
            if loss_inv == 'so3':
                loss = util.inv_err_randn(net, criterion, x, y_pred, y_true, 0.001, device)
            elif loss_inv == 'lorentz':
                loss = util.inv_err_lorentz(net, criterion, x, y_pred, y_true, 0.001, device)
            elif loss_inv == 'jet_center':
                loss = util.inv_err_px_py_pz_jet_center(net, criterion, x, y_pred, y_true, 0.001, device)
            elif loss_inv == 'phi':
                loss = util.inv_err_phi(net, criterion, x, y_pred, y_true, 0.001, device)
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
        inv_err = util.check_inv(net, x, device)
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        all_inv_err.append(inv_err)
        
        
        print(f'[{epoch + 1}] Trained loss: {train_loss:.5f}')
        print(f'[{epoch + 1}] Trained accuracy: {train_accu:.5f}')
        #print(f'[{epoch + 1}] Invariance relative error: {inv_err:.5f}')
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
    
    plt.plot(all_train_loss, color="gray", label="Training Loss")
    plt.plot(all_valid_loss, color="navy", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'plot/Loss_{name}_{data_name}')
    plt.close()
    plt.plot(all_train_accu, color="gray", label="Training Accuracy")
    plt.plot(all_valid_accu, color="navy", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f'plot/Accuracy_{name}_{data_name}')
    plt.close()
    plt.plot(all_inv_err, color="gray", label="Invariance Relative Error")
    plt.xlabel("Epochs")
    plt.ylabel("Invariance Relative Error")
    plt.savefig(f'plot/Invariance_{name}_{data_name}')
    print('Finished Training')


if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    main(config)