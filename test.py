import os
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--loss", type=str, default='default', choices=['default', 'so3', 'lorentz', 'jet_center', 'so3_eta_phi'], help = "Type of loss function to use")
parser.add_argument("--path", type=str, default='trained_models', help = "Name of folder just before accessing the config file and seed folder")
parser.add_argument("--pretrained", type = bool, default = False, help = "If the model is pretrained or not")
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
#torch.cuda.manual_seed_all(args.seed)


import yaml

from import_data import select_data
from import_model import get_model

import util
import plot_train

def main():
    name = args.name
    data_name = args.data_name
    loss_name = args.loss
    folder_name = args.path
    pretrained = args.pretrained
    seed = args.seed
    
    
    
    if pretrained:
        file_name = f"{name}_{data_name}"
        print(file_name)
        path = "pretrain_models/"
        
        model_file = path + f"trained_models_{file_name}/trained_models/seed{seed}/model_{file_name}.pt"
        config_path = path + f"trained_models_{file_name}/trained_models/config.yml"
    else:
        file_name = f"{name}_{data_name}_{loss_name}" if loss_name != 'default' else f"{name}_{data_name}"
        print(file_name)
        
        path = "train_models/"
    
        model_file = path + f"{file_name}/{folder_name}/seed{seed}/{file_name}.pt"
        config_path = path + f"{file_name}/{folder_name}/config.yml"

    
    config = util.load_config_file(config_path)
    print(f"Testing of the model {name} with data {data_name} using the config file {config_path}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing done on {device}")
    
    plotdir = f"plot/{file_name}"
    

    # Import the testing data
    _, test_data = select_data(name, data_name)
    nbr_const, nbr_feat = util.print_data_deets(test_data, test_data)
    print("")
    print(util.tcols.OKGREEN, "The Test data has been successfully imported", util.tcols.ENDC)
    print("")
    
    # Load the model from file
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"])
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    print("")
    print(util.tcols.OKGREEN, f"Model {name} loaded from {model_file} and moved to {device}", util.tcols.ENDC)
    print("")

    # Set the model to evaluation mode
    model.eval()    
    print(f'Validation of the model {name} trained using the data {data_name}\n')
    
    # Create the test data loader
    test_data = torch.utils.data.DataLoader(test_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=False, pin_memory = True, num_workers=1)
    x, _ = next(iter(test_data))
    
    # Check different properties of the model.    
    print(util.tcols.OKBLUE + "Model specification in terms of inference time, FLOPs and number of parameters" + util.tcols.ENDC)
    util.compute_inference_time(model, x, device)
    util.compute_flops(model, x, device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}" + "\n")
    plot_train.weights_plot(model, plotdir, file_name + '_test')
    print("\n" + util.tcols.OKBLUE + "Checking the invariance properties of the model..." + util.tcols.ENDC)
    if "px_py_pz_E" not in data_name:
        util.check_invariance(model, x, nbr_rot=32, device=device)
    else:
        util.check_invariance_Lorentz(model, x, nbr_rot=32, device=device)
    util.check_permutation_invariance(model, x, nbr_perm=32, device=device)
    print("\n")
    
    # Evaluate the model on the validation set
    evaluate_model(model, test_data, device)
    plot_evaluation_metric(name, model, test_data, plotdir, device = device)
    
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

def plot_evaluation_metric(name, model, validation_set, plotdir, device):
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
    util.roc_curves(plotdir, y_pred_all, y_true_all, roc_curve_name)
    util.dnn_output(plotdir, y_pred_all, name)

if __name__ == "__main__":

    main()