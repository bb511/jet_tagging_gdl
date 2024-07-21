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
    
    outdir = util.make_output_directory(f"trained_models_{name}", config["outdir"])
    # Save the config file to the main dir.
    util.save_config_file(config, outdir)
    config_path = os.path.join(f"trained_models_{name}_{data_name}/trained_models", f"config_{name}_{data_name}.yml")
    print('Config file saved')

    # Create subdir for each seed.
    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory(f"trained_models_{name}_{data_name}", config["outdir"])
    print('Subfile for each seeds are created')

    # Import the training and the validation data, sampled and normalised.
    train_data, valid_data = select_data(name, data_name)
    nbr_const, nbr_feat = util.print_data_deets(train_data, "Training")
    util.print_data_deets(valid_data, "Validation")
    print("The training data and the validation data have been successfully imported")

    # Import and train the model.
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"]).to(device)
    print('Model summary: ', model)
    #util.compute_inference_time(model, nbr_const, nbr_feat, name, device, config_path)
    #util.weights_plot(model, name + data_name + '_init')
    #if data_name not in ["px_py_pz_E_8", "px_py_pz_E_16", "px_py_pz_E_32", "px_py_pz_E_jet_center_8", "px_py_pz_E_jet_center_16", "px_py_pz_E_jet_center_32"]:
        #util.check_invariance(model, nbr_const, nbr_feat, name, device, config_path)
    #else:
        #util.check_invariance_Lorentz(model, nbr_const, nbr_feat, name, device, config_path)
    #util.check_permutation_invariance(model, nbr_const, nbr_feat, name, device, config_path)

    hist = train_model(model, train_data, valid_data, device, name, data_name, **config["training_hyperparams"], warmup_steps=1000)
    print('The model has been successfully trained')
    
    # Saved model and plot the loss and accuracy evolution throught the epochs.
    model_file = os.path.join(outdir, f"model_{name}_{data_name}.pt")
    torch.save(model.state_dict(), model_file)
    
    if config["validation"]:
        print(f'Validation of the model {name} trained using the data {data_name}')
        if data_name not in ["px_py_pz_E_8", "px_py_pz_E_16", "px_py_pz_E_32", "px_py_pz_E_jet_center_8", "px_py_pz_E_jet_center_16", "px_py_pz_E_jet_center_32"]:
            util.check_invariance(model, nbr_const, nbr_feat, name, device, config_path)
        else:
            util.check_invariance_Lorentz(model, nbr_const, nbr_feat, name, device, config_path)        
        util.check_permutation_invariance(model, nbr_const, nbr_feat, name, device, config_path)
        name = name + '_' + data_name
        util.weights_plot(model, name + '_validation')
        evaluate_model(model, valid_data, 'cuda:0', 256, config_path)
        plot_evaluation_metric(name, model, valid_data, batch_size=256, device = 'cuda:0')

    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(net, training_set, validation_set, device, name, data_name, epochs, lr, batch_size, loss_inv, warmup_steps):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)
    
    train_data = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    valid_data = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    
    total_steps = len(train_data) * epochs
    
    # Create scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    print(f'Beginning of training of model {name}')
    all_train_loss, all_train_accu, all_valid_loss, all_valid_accu, all_inv_err = [], [], [], [], []
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
        #inv_err = util.check_inv(net, device)
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        #all_inv_err.append(inv_err)
        
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
        
        if valid_accu > best_accu:
            best_accu = valid_accu
    
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
    #plt.plot(all_inv_err, color="gray", label="Invariance Relative Error")
    #plt.xlabel("Epochs")
    #plt.ylabel("Invariance Relative Error")
    #plt.savefig(f'plot/Invariance_{name}_{data_name}')
    print('Finished Training')
    
def evaluate_model(model, validation_set, device, batch_size, config_path):
    print('Evaluation of the model')
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()  
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, pin_memory = True)  
    accu = 0
    loss = 0
    num_batch = 0
    for x,y in validation_loader:
        x = x.to(device)
        y_true = y.to(device)
        
        y_pred = model(x)
        
        accu += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
        loss += criterion(y_pred, torch.argmax(y_true, dim=1)) 
        num_batch += 1

    print('Test Accuracy of the model : {} %'.format(accu/num_batch))
    print('The Test loss of the model : {} '.format(loss/num_batch))

def plot_evaluation_metric(name, model, validation_set, batch_size, device):
    model.eval()
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    for x,y in validation_loader:
        x = x.to(device)
        y_true = y.to(device)
        y_pred = nn.Softmax(dim = 1)(model(x))
    y_true = y_true.cpu().detach()
    y_pred = y_pred.cpu().detach()
    roc_curve_name = 'roc_curves_' + name
    util.roc_curves('plot', y_pred, y_true.numpy(), roc_curve_name)
    util.dnn_output('plot', y_pred, name)

if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    main(config)