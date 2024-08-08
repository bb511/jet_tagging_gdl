import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--config_file", type=str, help ="What are the configuration of your model", default= "config_jetclass.yml")
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

from import_data_jetclass import select_data
from import_model import get_model

import util
import pickle

def main(config):
    name = args.name
    data_name = args.data_name
    print(f"Training of the model {name} with data {data_name}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    outdir = util.make_output_directory(f"trained_models_{name}_{data_name}", config["outdir"])
    # Save the config file to the main dir.
    util.save_config_file(config, outdir)
    config_path = os.path.join(outdir, f"config_{name}_{data_name}.yml")
    print('Config file saved')

    # Create subdir for each seed.
    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory(f"trained_models_{name}_{data_name}", config["outdir"])
    print('Subfile for each seeds are created')

    # Import the training and the validation data, sampled and normalised.
    training_data, train_sampler, validation_data = select_data(name, data_name)
    
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=config["training_hyperparams"]["batch_size"], pin_memory = True, num_workers=12, sampler=train_sampler)#, collate_fn=custom_collate
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=False, pin_memory = True, num_workers=12)
    
    x, _ = next(iter(train_loader))
    print(f"Sample batch shape: {x.shape}")
    nbr_const, nbr_feat = x.shape[1], x.shape[2]
    # Import and train the model.
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"])
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    print('Model summary: ', model)
    util.compute_inference_time(model, x, name, device, config_path)
    util.weights_plot(model, name + data_name + '_init')
    if "px_py_pz_E" not in data_name:
        util.check_invariance(model, x, name, device, config_path)
    else:
        util.check_invariance_Lorentz(model, x, name, device, config_path)
    util.check_permutation_invariance(model, x, name, device, config_path)

    hist = train_model(model, train_loader, val_loader, device, name, data_name, **config["training_hyperparams"])
    print('The model has been successfully trained')
    
    # Saved model and plot the loss and accuracy evolution throught the epochs.
    model_file = os.path.join(outdir, f"model_{name}_{data_name}.pt")
    torch.save(model.state_dict(), model_file)
    
def custom_collate(batch):
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return features, labels

def train_model(net, train_data, valid_data, device, name, data_name, epochs, lr, batch_size, loss_inv):
    "Standard training function for pytorch model"
    l = 0.001
    criterion = nn.CrossEntropyLoss().to(device)
    if name not in ["vector_neuron", "vn_etaphi_rel", "vn_dgcnn"]:
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
    all_valid_loss = []
    all_valid_accu = []
    all_inv_err = []
    epochs_no_improve = 0
    epochs_es_limit = 50
    best_accu = 0
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    print("")
    next_epoch = 0
    prof = deepspeed.profiling.flops_profiler.FlopsProfiler(net)
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
            
            # get the inputs; data is a list of [inputs, labels]
            y_true = y
            if epoch == 0 and count == 0 and name != "cg_net":
                prof.start_profile()
                y_pred = net(x)
                prof.stop_profile()
                print(util.tcols.OKGREEN + "Model flops: " + util.tcols.ENDC, end="")
                print(f"{prof.get_total_flops()/ x.shape[0]}")
                prof.end_profile()
                print("")
                count += 1
            y_pred = net(x)
            if loss_inv =='so3':
                l = 0.001
                loss = util.inv_err_rand(net, criterion, x, y_pred, y_true, l, device)
            elif loss_inv == 'lorentz':
                l = 0.001
                loss = util.inv_err_lorentz(net, criterion, x, y_pred, y_true, l, device)
            elif loss_inv == 'jet_center':
                loss = util.inv_err_px_py_pz_jet_center(net, criterion, x, y_pred, y_true, l, device)
            elif loss_inv == 'so3_eta_phi':
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
        if "px_py_pz" in data_name and x.shape[2] == 3:
            inv_err = util.check_inv(net, x, device)
        else:
            inv_err = 100
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        all_inv_err.append(inv_err)
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
    plt.close()
    print('Finished Training')
    

if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    main(config)