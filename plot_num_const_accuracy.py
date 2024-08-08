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

import torch_geometric

import yaml
import deepspeed

from import_data import select_data
from import_model import get_model

import util
import pickle

def main(config):
    name = args.name
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
    outdir = util.make_output_directory(f"value_accuracy_vs_nconst", config["outdir"])
    print('Subfile for each seeds are created')

    # Import the training and the validation data, sampled and normalised.
    training_data, test_data = select_data(name, data_name)
    util.print_data_deets(training_data, test_data)
    print("The training data and the validation data have been successfully imported")
    nbr_const = training_data[:][0].shape[1]
    nbr_feat = training_data[:][0].shape[2]
    train_load = torch.utils.data.DataLoader(training_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory = True, num_workers=1)
    valid_load = torch.utils.data.DataLoader(test_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory = True, num_workers=1)

    # Import and train the model.
    accuracies = []
    nbr_pts = []
    for i in range(28,33, 4):#
        if i == 0:
            i = 1
        print(f'Training the model with {i} constituents')
        model = get_model(name, i, nbr_feat, config["model_hyperparameter"]).to(device)
        model.apply(util.init_weights)

        hist = train_model(model, train_load, device, name, i,  **config["training_hyperparams"])
        accu = evaluate_model(model, valid_load, 'cuda:0', 64, config_path, i)
        accuracies.append(accu.cpu().numpy())
        nbr_pts.append(i)
        del model
    print(accuracies)
    np.save(outdir + f'/accuracy_vs_constituents_{name}_{data_name}_bis.npy', accuracies)
    np.save(outdir + f'nbr_pts_{name}_{data_name}.npy', nbr_pts)
    print('The model has been successfully trained')
    plt.figure(figsize=(10, 6))
    plt.plot(nbr_pts, accuracies, marker='o')
    plt.xlabel('Number of Constituents')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Number of Constituents')
    plt.savefig(f'./plot/accuracy_vs_constituents_{name}_{data_name}.png')
    plt.show()
    


def train_model(net, train_data, device, name, i,  epochs, lr, batch_size, loss_inv):
    "Standard training function for pytorch model"
    l = 0.001
    criterion = nn.CrossEntropyLoss().to(device)
    if name not in ["vector_neuron", "vn_dgcnn"]:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr = 0.0294)#
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
    for epoch in range(epochs):  # loop over the dataset multiple times
        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        net.train()
        print(f'Begining of Epoch {epoch + 1}/{epochs}')
        count = 0
        for x,y in train_data:
            x = x.to(device)
            y = y.to(device)
            
            x = x[:, :i, :]
            
            # get the inputs; data is a list of [inputs, labels]
            y_true = y
            y_pred = net(x)
            if loss_inv =='so3':
                l = 0.001
                loss = util.inv_err_rand(net, criterion, x, y_pred, y_true, l, device)
            elif loss_inv == 'lorentz':
                l = 0.001
                loss = util.inv_err_lorentz(net, criterion, x, y_pred, y_true, l, device)
            elif loss_inv == 'jet_center':
                loss = util.inv_err_px_py_pz_jet_center(net, criterion, x, y_pred, y_true, l, device)
            elif loss_inv == 'phi':
                loss = util.inv_err_phi(net, criterion, x, y_pred, y_true, l, device)
            elif loss_inv == "so3_eta_phi":
                l = 0.001
                loss = util.inv_err_rand_eta_phi(net, criterion, x, y_pred, y_true, l, device)
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
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        print(f'[{epoch + 1}] Trained loss: {train_loss:.5f}')
        print(f'[{epoch + 1}] Trained accuracy: {train_accu:.5f}')
        
    
def evaluate_model(model, validation_loader, device, batch_size, config_path, i):
    print('Evaluation of the model')
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()  
    accu = 0
    loss = 0
    num_batch = 0
    for x,y in validation_loader:
        x = x.to(device)
        y_true = y.to(device)
        
            
        x = x[:, :i, :]
        
        y_pred = model(x)
        
        accu += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
        loss += criterion(y_pred, torch.argmax(y_true, dim=1)) 
        num_batch += 1

    print('Test Accuracy of the model : {} %'.format(accu/num_batch))
    print('The Test loss of the model : {} '.format(loss/num_batch))
    return accu/num_batch


if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    main(config)