import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
import yaml
import deepspeed

# Import the model from a separate file
from import_model import get_model
from import_data import select_data

import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--config_file", type=str, help="Configuration of your model", default="config_equivariant.yml")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main(config):
    name = args.name
    assert name == "graph_net", "Only graph_net model is supported for now"
    data_name = args.data_name
    print(f"Training of the model {name} with data {data_name}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    outdir = util.make_output_directory(f"trained_models_{name}", config["outdir"])
    util.save_config_file(config, outdir)
    config_path = os.path.join(f"trained_models_{name}_{data_name}/trained_models", f"config_{name}_{data_name}.yml")
    print('Config file saved')

    config["outdir"] = os.path.join(config["outdir"], f"seed{args.seed}")
    outdir = util.make_output_directory(f"trained_models_{name}_{data_name}", config["outdir"])
    print('Subfile for each seed is created')

    # Import the training and the validation data
    train_data, valid_data = select_data(name, data_name)
    nbr_const, nbr_feat = 8, 3#util.print_data_deets(train_data, "Training")
    print("The training data and the validation data have been successfully imported")

    
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"]).to(device)
    print('Model summary: ', model)
    
    #util.compute_inference_time(model, device, config_path)
    #util.weights_plot(model, name + data_name + '_init')

    hist = train_model(model, train_data, valid_data, device, name, data_name, **config["training_hyperparams"])
    print('The model has been successfully trained')
    
    # Save model and plot the loss and accuracy evolution through the epochs
    model_file = os.path.join(outdir, f"model_{name}_{data_name}.pt")
    torch.save(model.state_dict(), model_file)
    
    if config["validation"]:
        print(f'Validation of the model {name} trained using the data {data_name}')
        name = name + '_' + data_name
        util.weights_plot(model, name + '_validation')
        evaluate_model(model, valid_data, device, 256, config_path)
        plot_evaluation_metric(name, model, valid_data, batch_size=256, device=device)
        
def masked_loss(output, target, mask):
    loss = F.cross_entropy(output, target, reduction='none').to(mask.device)
    mask = mask.float()
    mask /= mask.mean()
    loss *= mask
    return loss.mean()

def train_model(net, training_set, validation_set, device, name, data_name, epochs, lr, batch_size, loss_inv):
    #criterion = masked_loss#.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=80, factor=0.1, threshold=1e-4, verbose=True)
    
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    
    print(f'Beginning of training of model {name}')
    all_train_loss = []
    all_train_accu = []
    all_valid_loss = []
    all_valid_accu = []
    
    for epoch in range(epochs):
        net.train()
        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            y_pred, mask = net(x.x, x.edge_index, x.batch)
            y_true = y
            
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            batch_loss_sum += loss.detach()
            totnum_batches += 1
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
        
        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        
        print(f'[{epoch + 1}] Trained loss: {train_loss:.5f}')
        print(f'[{epoch + 1}] Trained accuracy: {train_accu:.5f}')
        
        net.eval()
        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        
        with torch.no_grad():
            for x,y in valid_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred, mask = net(x.x, x.edge_index, x.batch)
                y_true = y
                
                loss = criterion(y_pred, y_true)
                
                batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
                batch_loss_sum += loss.detach()
                totnum_batches += 1 
        
        valid_loss = batch_loss_sum / totnum_batches
        valid_accu = batch_accu_sum / totnum_batches
        all_valid_loss.append(valid_loss.item())
        all_valid_accu.append(valid_accu.item())
        
        print(f"[{epoch + 1}] validation loss: {valid_loss:.5f}")
        print(f"[{epoch + 1}] test accuracy: {valid_accu:.5f}")
        print(f'[{epoch + 1}] Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
        
        scheduler.step(valid_accu)

    # Plot training and validation curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_train_loss, label="Training Loss")
    plt.plot(all_valid_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(all_train_accu, label="Training Accuracy")
    plt.plot(all_valid_accu, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(f'plot/Training_Curves_{name}_{data_name}.png')
    plt.close()
    
    print('Finished Training')

def evaluate_model(model, validation_set, device, batch_size, config_path):
    print('Evaluation of the model')
    criterion = masked_loss#.to(device)
    model.eval()  
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    accu = 0
    loss = 0
    num_batch = 0
    
    with torch.no_grad():
        for x,y in validation_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred, mask = model(x.x, x.edge_index, x.batch)
            y_true = y
            
            accu += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
            loss += criterion(y_pred, y_true, mask)
            num_batch += 1

    print('Test Accuracy of the model : {} %'.format(accu/num_batch * 100))
    print('The Test loss of the model : {} '.format(loss/num_batch))

def plot_evaluation_metric(name, model, validation_set, batch_size, device):
    model.eval()
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    all_y_pred = []
    all_y_true = []
    
    with torch.no_grad():
        for x,y in validation_loader:
            x = x.to(device)
            y = y.to(device)   
            y_pred = nn.Softmax(dim=1)(model(x.x, x.edge_index, x.batch))
            all_y_pred.append(y_pred.cpu())
            all_y_true.append(y.cpu())
    
    y_pred = torch.cat(all_y_pred, dim=0)
    y_true = torch.cat(all_y_true, dim=0)
    
    roc_curve_name = 'roc_curves_' + name
    util.roc_curves('plot', y_pred, y_true.numpy(), roc_curve_name)
    util.dnn_output('plot', y_pred, name)

if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    main(config)