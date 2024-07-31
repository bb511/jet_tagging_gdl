import os
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
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


import yaml

from import_data import select_data
from import_model import get_model

import util

def main():
    name = args.name
    data_name = args.data_name
    model_file = f"train_models/trained_models_{name}_{data_name}/trained_models/seed42/model_{name}_{data_name}.pt"
    config_path = f"train_models/trained_models_{name}_{data_name}/trained_models/config.yml"

    
    config = util.load_config_file(config_path)
    print(f"Testing of the model {name} with data {data_name} using the config file {config_path}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing done on {device}")
    

    # Import the testing data
    training_data, test_data = select_data(name, data_name)
    nbr_const, nbr_feat = util.print_data_deets(training_data, test_data)
    print("")
    print(util.tcols.OKGREEN, "The Test data has been successfully imported", util.tcols.ENDC)
    print("")
    
    # Load the model from file
    model =  get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"])
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    print("")
    print(util.tcols.OKGREEN, f"Model {name} loaded from {model_file} and moved to {device}", util.tcols.ENDC)
    print("")

    # Evaluate the model on the validation set
    model.eval()    
    print(f'Validation of the model {name} trained using the data {data_name}\n')
    test_data = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, pin_memory = True, num_workers=1)
    x, _ = next(iter(test_data))
    util.compute_inference_time(model, x, name, device, config_path)
    if "px_py_pz_E" not in data_name:
        util.check_invariance(model, x, name, device, config_path)
    else:
        util.check_invariance_Lorentz(model, x, name, device, config_path)        
    util.check_permutation_invariance(model, x, name, device, config_path)
    name = name + '_' + data_name
    util.weights_plot(model, name + '_test')
    evaluate_model(model, test_data, 'cuda:0')
    plot_evaluation_metric(name, model, test_data, device = 'cuda:0')
    
def evaluate_model(model, validation_set, device):
    print('Evaluation of the model\n')
    criterion = nn.CrossEntropyLoss().to(device)
    accu = 0
    loss = 0
    num_batch = 0
    for x,y in validation_set:
        x = x.to(device)
        y_true = y.to(device)
        
        y_pred = model(x)
        
        accu += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
        loss += criterion(y_pred, torch.argmax(y_true, dim=1)).item()
        num_batch += 1

    print(util.tcols.OKBLUE,'Test Accuracy of the model : {} %'.format(accu/num_batch), util.tcols.ENDC, '\n')
    print('The Test loss of the model : {} '.format(loss/num_batch))

def plot_evaluation_metric(name, model, validation_set, device):
    y_true_all = []
    y_pred_all = []
    for x,y in validation_set:
        x = x.to(device)
        y_true = y.to(device)
        y_pred = nn.Softmax(dim = 1)(model(x))
        y_true = y_true.cpu().detach()
        y_pred = y_pred.cpu().detach()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
    roc_curve_name = 'roc_curves_' + name
    y_pred_all = torch.cat(y_pred_all, dim = 0)
    y_true_all = torch.cat(y_true_all, dim = 0)
    util.roc_curves('plot', y_pred_all, y_true_all, roc_curve_name)
    util.dnn_output('plot', y_pred_all, name)

if __name__ == "__main__":

    main()