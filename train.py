import os
import argparse
import yaml
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--config_file", type=str, help ="What are the configuration of your model", default= "config_equivariant.yml")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--loss", type=str, default='default', choices=['default','so2', 'so3', 'lorentz', 'jet_center','so2_eta_phi', 'so3_eta_phi'], help = "Type of loss function to use")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
parser.add_argument("--fold", type=int, default=None, help="Fold number for cross-validation (default: None)")
parser.add_argument("--lam", type=float, default=0.001, help="Regularization weight λ, in case of using the symmetric loss")
parser.add_argument("--nr_rot", type=int, default=8, help="Number of rotation matrices used for data augmentation")
parser.add_argument("--outdir", type=str, default=None, help="Output directory for saving results, in case you want in a different file than in the config")
parser.add_argument("--dataset_size", type=int, default=0, help="Size of the dataset to use for training. If 0, use the full dataset.")
#parser.add_argument("--norm_outside", type=bool, default=True)
#parser.add_argument("--device", type=str, default='cuda:0')
args = parser.parse_args()

import random
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim

from data_extraction import DataSelection
from import_model import get_model
import data.standardization as standardization
import data.plot_data as plot_data
import util
import plot_train
from train_helpers import train_model, get_cosine_schedule_with_warmup
import custom_loss as custom_loss

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
######################################################################################################################################

def main(config):
    name = args.name
    data_name = args.data_name
    loss_name = args.loss
    
    # Set up the file name
    file_name = f"{name}_{data_name}_{loss_name}" if loss_name != 'default' else f"{name}_{data_name}"

    print(f"Training of the model {name} with data {data_name} and using loss {loss_name}") 
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training done on {device}")
    
    if args.fold == None:
        if args.outdir is None:
            plotdir = util.make_output_directory(f"plot/{file_name}/", config["outdir"])
            outdir = util.make_output_directory(f"train_models/{file_name}/", config["outdir"])
        else:
            plotdir = util.make_output_directory(f"plot/{file_name}/", args.outdir)
            outdir = util.make_output_directory(f"train_models/{file_name}/", args.outdir)
            config["outdir"] = args.outdir
    else:
        plotdir = util.make_output_directory(f"plot_folds/{file_name}/", config["outdir"])
        outdir = util.make_output_directory(f"train_folds/{file_name}/", config["outdir"])
    
    # Save the config file to the main dir.
    util.save_config_file(config, outdir)
    config_path = os.path.join(outdir, f"config_{file_name}.yml")
    print(f'Config file saved in {config_path}')
    
    if args.fold is not None:
        fold_dir = os.path.join(config["outdir"], f"fold{args.fold}")
        outdir = util.make_output_directory(f"train_folds/{file_name}/", fold_dir)
        plotdir = util.make_output_directory(f"plot_folds/{file_name}/", fold_dir)

    # Import the training and the validation data
    print(util.tcols.OKBLUE + "Importing data..." + util.tcols.ENDC)
    
    data_select = DataSelection(
        data_name=data_name, train = True,
        dataset_size=args.dataset_size,
        split_seed=config["data_hyperparams"].get("split_seed", 12),
        split_ratio=config["data_hyperparams"].get("split_ratio", 0.8),
        total_folds=config["data_hyperparams"].get("total_folds", 5),
        augment_seed=config["data_hyperparams"].get("augment_seed", 15),
        augmentation_type=config["data_hyperparams"].get("augmentation_type", 'scipy'),
        augmentation_space=config["data_hyperparams"].get("augmentation_space", 'so3'),
        nbr_rot = args.nr_rot,
        norm_type=config["data_hyperparams"].get("norm", 'robust'),
        norm_outside=config["data_hyperparams"].get("norm_outside", True)
                               )
    
    x_train, y_train, jet_train, x_val, y_val, jet_val = data_select.get_train_data(fold_number=args.fold)
    print("Training on fold ", args.fold)
    
    parts = data_name.split('_')
    filtered_parts = [part for part in parts[:-1] if part not in ['data', 'aug']]
    feats = '_'.join(filtered_parts)
    plot_data.normalised_data_subplot(Path(plotdir), x_train, y_train, "train", feats)
    plot_data.normalised_data_subplot(Path(plotdir), x_val, y_val, "val", feats)
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train, jet_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val, jet_val)
    nbr_const = train_dataset[:][0].shape[1]
    nbr_feat = train_dataset[:][0].shape[2]
    
    print(util.tcols.OKGREEN + "Data imported." + util.tcols.ENDC)
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Training and validation data loaders
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory = True, num_workers=1)
    valid_load = torch.utils.data.DataLoader(val_dataset, batch_size=config["training_hyperparams"]["batch_size"], shuffle=True, pin_memory = True, num_workers=1)
        
    # Compute and Save the normalization coefficients.
    raw_coef_norm = data_select.norm_coef
    
    if config["data_hyperparams"]["norm_outside"] == False:
        coef_norm_path = os.path.join(outdir, f"coef_norm.json")
        standardization.save_normalization_coeff(raw_coef_norm, coef_norm_path)
        coef_norm = standardization.load_normalization_coeff(data_name, coef_norm_path)
    else:
        coef_norm_path = os.path.join(outdir, f"coef_norm.pkl")
        with open(coef_norm_path, "wb") as f:
            pickle.dump(raw_coef_norm, f)
        coef_norm = None

    # Create subdir for each seed.  DISABLED FOR NOW
    # if args.fold is None:
    #     seed_dir = os.path.join(config["outdir"], f"seed{args.seed}")
    #     outdir = util.make_output_directory(f"train_models/{file_name}", seed_dir)
    # print('Subfile for each seeds are created')

####################################################################################################################

    # Import the model.
    print("")
    print(util.tcols.OKBLUE + "Importing model..." + util.tcols.ENDC)
    model = get_model(name, nbr_const, nbr_feat, coef_norm, config["model_hyperparameter"]).to(device) 
    model.apply(util.init_weights)
    
    print(util.tcols.OKCYAN + 'Model summary: ' + util.tcols.ENDC, '\n', model)
    
    # Check different properties of the model.
    x, _, _= next(iter(train_load))
    
    print("\n")
    print(util.tcols.OKBLUE + "Model specification in terms of inference time, FLOPs and number of parameters" + util.tcols.ENDC)
    util.compute_inference_time(model, x, device)
    util.compute_flops(model, x, device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print("\n")
    plot_train.weights_plot(model, plotdir, file_name + '_init')
    print("\n")
    print(util.tcols.OKBLUE + "Checking the invariance properties of the model..." + util.tcols.ENDC)
    
    # if "px_py_pz" in data_name:
    #     if config["data_hyperparams"]["augmentation_space"] == 'so3':
    #         init_inv_err = custom_loss.inv_err_so3(model, None, x, y_pred, y_true, jet, coef_norm, l, rot_type, 16, device, filters, return_only_error=True))
    #     if "px_py_pz_E" in data_name:
    #         util.check_invariance_Lorentz(model, x, nbr_rot=32, device=device)
        
    custom_loss.check_permutation_invariance(model, x, nbr_perm=32, device=device)
    print("\n")
    
######################################################################################################################################
    
    criterion = nn.CrossEntropyLoss().to(device)
    lr = config["training_hyperparams"]["lr"]
    
    if name not in ["vector_neuron", "vn_etaphi_rel", "vn_dgcnn", 'transformer', "vector_neuron_so2"]:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    elif name == "transformer":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    print('Optimizer Summary: ', optimizer)
    
    if name == "transformer":
        warmup_steps = 1000
        epochs = config["training_hyperparams"]["epochs"]
        total_steps = len(train_load) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
    elif name in ["vector_neuron", "vector_neuron_so2"]:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode="max",
                                                                patience= 80,
                                                                factor=0.1,
                                                                threshold=1e-4,
                                                                verbose=True,
                                                                )
    
    # In this case: normalization and rotation are non commutative
    if config["data_hyperparams"]["norm"] == 'minmaxcustom' and loss_name in ['so2', 'so2_eta_phi']:
        raw_coef_norm = None
    
    rot_type = config["data_hyperparams"].get("augmentation_type", 'scipy')
    rot_space = config["data_hyperparams"].get("augmentation_space", 'so2')
    filters = config["data_hyperparams"].get("filters", {"pt":(0,1600), "eta":(-3,3), "etarel":(-0.5,0.5), "phirel":(-0.5, 0.5)})
    compare_to_label = config["data_hyperparams"].get("compare_to_label", False)
    
    print(f"Rotation type: {rot_type}, Rotation space: {rot_space}")
    print(f"Filters: {filters}")
    
    hist = train_model(model, train_load, valid_load, device, criterion, optimizer, scheduler, 
                       name, data_name, loss_name, rot_type, rot_space, filters, compare_to_label,
                       outdir, file_name, plotdir, raw_coef_norm,
                       **config["training_hyperparams"], l=args.lam)
    
    print('The model has been successfully trained')

    
######################################################################################################################################

if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    main(config)