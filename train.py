import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--config_file", type=str, help ="What are the configuration of your model", default= "config_equivariant.yml")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--loss", type=str, default='default', choices=['default', 'so3', 'lorentz', 'jet_center', 'so3_eta_phi'], help = "Type of loss function to use")
parser.add_argument("--seed", type=int, default=42)
#parser.add_argument("--device", type=str, default='cuda:0')
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

from import_data import select_data
from import_model import get_model

import custom_loss

import util
import plot_train

def main(config):
    name = args.name
    assert name != "transformer", "The transformer model is implemented in the script train_transformer.py"
    data_name = args.data_name
    loss_name = args.loss
    
    # Set up the file name
    file_name = f"{name}_{data_name}_{loss_name}" if loss_name != 'default' else f"{name}_{data_name}"

    print(f"Training of the model {name} with data {data_name} and using loss {loss_name}") 
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    plotdir = util.make_output_directory("plot", file_name)
    outdir = util.make_output_directory(f"train_models/{file_name}/", config["outdir"])
    
    # Save the config file to the main dir.
    util.save_config_file(config, outdir)
    config_path = os.path.join(outdir, f"config_{file_name}.yml")
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
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"]).to(device)
    model.apply(util.init_weights)
    
    print(util.tcols.OKCYAN + 'Model summary: ' + util.tcols.ENDC, '\n', model)
    
    # Check different properties of the model.
    x, _ = next(iter(train_load))
    
    print("\n")
    print(util.tcols.OKBLUE + "Model specification in terms of inference time, FLOPs and number of parameters" + util.tcols.ENDC)
    util.compute_inference_time(model, x, device)
    util.compute_flops(model, x, device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print("\n")
    plot_train.weights_plot(model, plotdir, file_name + '_init')
    print("\n")
    print(util.tcols.OKBLUE + "Checking the invariance properties of the model..." + util.tcols.ENDC)
    if "px_py_pz_E" not in data_name:
        util.check_invariance(model, x, nbr_rot=32, device=device)
    else:
        util.check_invariance_Lorentz(model, x, nbr_rot=32, device=device)
    util.check_permutation_invariance(model, x, nbr_perm=32, device=device)
    print("\n")

    hist = train_model(model, train_load, valid_load, device, name, data_name, loss_name, plotdir, **config["training_hyperparams"])
    print('The model has been successfully trained')
    
    # Saved model and plot the loss and accuracy evolution throught the epochs.3
    model_file = os.path.join(outdir, f"{file_name}.pt")
    torch.save(model.state_dict(), model_file)
    print(f'Model saved at {model_file}')


def train_model(net, train_data, valid_data, device, name, data_name, loss_inv, plotdir, epochs, batch_size, lr, l = 0.001):
    "Standard training function for pytorch model"
    print(util.tcols.OKCYAN + f"Training of the model {name} with data {data_name} and using loss {loss_inv} with the following hyperparameters: epochs = {epochs}, batch_size = {batch_size}, learning rate = {lr}" + util.tcols.ENDC + '\n')
    criterion = nn.CrossEntropyLoss().to(device)
    if name not in ["vector_neuron", "vn_etaphi_rel", "vn_dgcnn"]:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr) #get better performance with RMSprop for VN
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode="max",
                                                            patience= 80,
                                                            factor=0.1,
                                                            threshold=1e-4,
                                                            verbose=True,
                                                            )
    print('Optimizer Summary: ', optimizer)
    print(f'Begining of training of model {name}')
    all_train_loss = []
    all_train_accu = []
    all_valid_loss = []
    all_valid_accu = []
    all_inv_err = []
    epochs_no_improve = 0
    epochs_es_limit = 50
    best_accu = 0
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        net.train()
        print(f'Begining of Epoch {epoch + 1}/{epochs}')
        for x,y_true in train_data:
            x, y_true = x.to(device), y_true.to(device)
            
            y_pred = net(x)
            
            # compute the loss function based on the user choice
            if loss_inv =='so3':
                loss = custom_loss.inv_err_rand(net, criterion, x, y_pred, y_true, l, 16, device)
            elif loss_inv == 'lorentz':
                loss = custom_loss.inv_err_lorentz(net, criterion, x, y_pred, y_true, l, 16, device)
            elif loss_inv == 'jet_center':
                loss = custom_loss.inv_err_px_py_pz_jet_center(net, criterion, x, y_pred, y_true, l, 16, device)
            elif loss_inv == 'so3_eta_phi':
                #loss = custom_loss.inv_err_deterministic_eta_phi(net, criterion, x, y_pred, y_true, l, 16, device)
                #loss = custom_loss.inv_err_sobol_eta_phi(net, criterion, x, y_pred, y_true, l, 16, device)
                loss = custom_loss.inv_err_rand_eta_phi(net, criterion, x, y_pred, y_true, l, 16, device)
            else:
                loss = criterion(y_pred, torch.argmax(y_true, dim = 1))
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            batch_loss_sum += loss.detach()
            totnum_batches += 1
            batch_accu_sum += torch.sum(
                y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]
            ) / len(y_true)
            
        # print statistics
        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches
        if "px_py_pz" in data_name:
            inv_err = util.check_inv(net, x, device)
        else:
            inv_err = 100
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        all_inv_err.append(inv_err.item())
        print(f'[{epoch + 1}] Trained loss: {train_loss:.5f}')
        print(f'[{epoch + 1}] Trained accuracy: {train_accu:.5f}')
        print(f'[{epoch + 1}] Invariance relative error: {inv_err:.5f}')
        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        
        with torch.no_grad():
            net.eval()
            for x,y in valid_data:
                x = x.to(device)
                y = y.to(device)        

                y_pred = net(x)
                y_true = y
                
                loss = criterion(y_pred, torch.argmax(y_true, dim = 1).to(torch.int64))
                
                batch_accu_sum += torch.sum(
                    y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]
                    ) / len(y_true)
                batch_loss_sum += loss.detach()
                
                totnum_batches += 1 
            valid_loss = batch_loss_sum / totnum_batches
            valid_accu = batch_accu_sum / totnum_batches
            print(f"[{epoch + 1}] validation loss: {valid_loss:.5f}")
            print(f"[{epoch + 1}] test accuracy: {valid_accu:.5f}")
            print(f'[{epoch + 1}] Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
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