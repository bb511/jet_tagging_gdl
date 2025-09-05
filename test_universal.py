import os
import argparse
import yaml
import json
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--loss", type=str, default='default', choices=['default','so2', 'so3', 'lorentz', 'jet_center','so2_eta_phi', 'so3_eta_phi'], help = "Type of loss function to use")
parser.add_argument("--path", type=str, default='trained_models', help = "Name of folder just before accessing the config file and seed folder")
#parser.add_argument("--pretrained", type = bool, default = False, help = "If the model is pretrained or not")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--folds", type=bool, default=False)
#parser.add_argument("--device", type=str, default='cuda:0')
args = parser.parse_args()

import random
#random.seed(args.seed)
import numpy as np
#np.random.seed(args.seed)
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed_all(args.seed)

from data_extraction import DataSelection
from import_model_reorg import get_model
import data.standardization as standardization
import data.plot_data as plot_data

import custom_loss_new as custom_loss
import util
import plot_train
from test_helpers import evaluate_model, plot_evaluation_metric, convert_value

######################################################################################################################################
######################################################################################################################################

def main():
    name = args.name
    data_name = args.data_name
    loss_name = args.loss
    folder_name = args.path
    #pretrained = args.pretrained
    seed = args.seed
    
    # if pretrained:
    #     file_name = f"{name}_{data_name}"
    #     print(file_name)
    #     path = "pretrain_models/"
        
    #     model_file = path + f"trained_models_{file_name}/trained_models/seed{seed}/model_{file_name}.pt"
    #     config_path = path + f"trained_models_{file_name}/trained_models/config.yml"
    # else:
    
    file_name = f"{name}_{data_name}_{loss_name}" if loss_name != 'default' else f"{name}_{data_name}"
    print(file_name)
    
    if not args.folds:
        path = "train_models/"
        config_path = path + f"{file_name}/{folder_name}/config.yml"
        config = util.load_config_file(config_path)
        
        plotdir_model_file = f"plot/{file_name}/{config['outdir']}"
        plotdirs = [f"plot/{file_name}/{config['outdir']}"]
        
        if config["data_hyperparams"]["norm_outside"]:
            coef_norm_path = path + f"{file_name}/{folder_name}/coef_norm.pkl"
            coef_norms = [pickle.load(open(coef_norm_path, "rb"))]
        else:
            coef_norm_path = path + f"{file_name}/{folder_name}/coef_norm.json"
            coef_norms = [standardization.load_normalization_coeff(data_name, coef_norm_path)]
            
        #model_files = [path + f"{file_name}/{folder_name}/seed{seed}/{file_name}.pt"]
        model_files = [path + f"{file_name}/{folder_name}/{file_name}_best_val.pt"]
        
    else:
        path = "train_folds/"
        config_path = path + f"{file_name}/{folder_name}/config.yml"
        config = util.load_config_file(config_path)
        total_folds = config["data_hyperparams"].get("total_folds", 5)
        
        plotdir_model_file = f"plot_folds/{file_name}/{config['outdir']}"
        plotdirs = [f"plot_folds/{file_name}/{config['outdir']}/fold{fold}" for fold in range(total_folds)]
        
        if config["data_hyperparams"]["norm_outside"]:
            coef_norm_paths = [path + f"{file_name}/{folder_name}/fold{fold}/coef_norm.pkl" for fold in range(total_folds)]
            coef_norms =[pickle.load(open(coef_norm_path, "rb")) for coef_norm_path in coef_norm_paths]      
        else:
            coef_norm_paths = [path + f"{file_name}/{folder_name}/fold{fold}/coef_norm.json" for fold in range(total_folds)]
            coef_norms = [standardization.load_normalization_coeff(data_name, coef_norm_path) for coef_norm_path in coef_norm_paths]
            
        model_files = [path + f"{file_name}/{folder_name}/fold{fold}/{file_name}_best_val.pt" for fold in range(total_folds)]

    print(f"Testing of the model {name} with data {data_name} using the config file {config_path}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing done on {device}")
    
    inf_times = []
    models_flops = []
    nums_params = []
    mean_inv_errs = []
    perm_invs = []
    test_accus = []
    test_losses = []
    aucss = []
    
    for i, (plotdir, model_file, coef_norm) in enumerate(zip(plotdirs, model_files, coef_norms)):
        
        if args.folds:
            print(f"Evaluation on fold: {i}")
        
        data_select = DataSelection(
            data_name=data_name, train = False,
            norm_type=config["data_hyperparams"].get("norm", 'robust'),
            norm_outside=config["data_hyperparams"].get("norm_outside", True),
            norm_coef = coef_norm
                                    )
        x_test, y_test, jet_test = data_select.get_data()
        parts = data_name.split('_')
        filtered_parts = [part for part in parts[:-1] if part not in ['data', 'aug']]
        feats = '_'.join(filtered_parts)
        plot_data.normalised_data_subplot(Path(plotdir), x_test, y_test, "test", feats)
        
        test_data = torch.utils.data.TensorDataset(x_test, y_test, jet_test)
        nbr_const, nbr_feat = util.print_data_deets(test_data, "Test")
        
        print("")
        print(util.tcols.OKGREEN, "The Test data has been successfully imported", util.tcols.ENDC)
        print("")
        
        # if config["data_hyperparams"]["norm_outside"]:
        #     coef_norm = None
        # Load the model from file
        print("")
        print(util.tcols.OKBLUE + "Importing model..." + util.tcols.ENDC)
        
        if config["data_hyperparams"]["norm_outside"]:
            model = get_model(name, nbr_const, nbr_feat, None, config["model_hyperparameter"]).to(device) 
        else:
            model = get_model(name, nbr_const, nbr_feat, coef_norm, config["model_hyperparameter"]).to(device)
        
        # print("phi layers:")
        # for j, layer in enumerate(model.phi):
        #     print(f"  phi[{j}]: {layer}")

        # print("rho layers:")
        # for j, layer in enumerate(model.rho):
        #     print(f"  rho[{j}]: {layer}")

        # try:
        #     model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        # except RuntimeError as e:
        #     print(f"Skipping fold {i} due to model mismatch: {e}")
        #     continue
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        
        model.to(device)
        print("")
        print(util.tcols.OKGREEN, f"Model {name} loaded from {model_file} and moved to {device}", util.tcols.ENDC)
        print("")

        # Set the model to evaluation mode
        model.eval()    
        print(f'Evaluation of the model {name} trained using the data {data_name}\n')
        
        # Create the test data loader
        test_data = torch.utils.data.DataLoader(test_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=False, pin_memory = True, num_workers=1)
        x, _, jet = next(iter(test_data))
        x = x.to(device)
        jet = jet.to(device)
        
        # Check different properties of the model.    
        print(util.tcols.OKBLUE + "Model specification in terms of inference time, FLOPs and number of parameters" + util.tcols.ENDC)
        
        inf_time = util.compute_inference_time(model, x, device)
        inf_times.append(inf_time)
        
        model_flops = util.compute_flops(model, x, device)
        models_flops.append(model_flops)
        
        num_params = sum(p.numel() for p in model.parameters())
        nums_params.append(num_params)
        
        print(f"Number of parameters: {num_params}" + "\n")
        
        plot_train.weights_plot(model, plotdir, file_name + '_test')
        
        print("\n" + util.tcols.OKBLUE + "Checking the invariance properties of the model..." + util.tcols.ENDC)
        if "px_py_pz" in data_name:
            rot_type = config["data_hyperparams"].get("augmentation_type", 'scipy')
            rot_space = config["data_hyperparams"].get("augmentation_space", 'so2')
            filters = config["data_hyperparams"].get("filters", {"pt":(0,1600), "eta":(-3,3), "etarel":(-0.5,0.5), "phirel":(-0.5, 0.5)})
            
            if config["data_hyperparams"]["norm"] == 'minmaxcustom' and loss_name in ['so2', 'so2_eta_phi']:
                coef_norm = None
                
            print(f"Rotation type: {rot_type}, Rotation space: {rot_space}, Filters: {filters}")
            if rot_space == "so3":
                mean_inv_err = custom_loss.check_so3_rot_err(model, x, jet, coef_norm, rot_type, 32, device, filters)
            elif rot_space == "so2":
                mean_inv_err = custom_loss.check_so2_rot_err(model, x, coef_norm, 32, device)
        else:
            mean_inv_err = float('nan')
            # mean_inv_err = util.check_invariance_Lorentz(model, x, nbr_rot=32, device=device)
        print(f"Mean invariance error: {mean_inv_err:.4f}%")
        mean_inv_errs.append(mean_inv_err)
        
        perm_inv = custom_loss.check_permutation_invariance(model, x, nbr_perm=32, device=device)
        perm_invs.append(perm_inv)
        print("\n")
        
        # Evaluate the model on the validation set
        test_accu, test_loss = evaluate_model(model, test_data, device)
        test_accus.append(test_accu)
        test_losses.append(test_loss)
        
        aucs = plot_evaluation_metric(name, model, test_data, plotdir, device = device)
        aucss.append(aucs)
        del model
    
####################################################################################################################
    
    labels = ["Quark", "Gluon", "W", "Z", "Top"]
    # Create a dictionary with all model characteristics
    
    if not args.folds:
        model_info = {
            "Model Specifications": {
                "Inference Time (ms)": convert_value(inf_time),
                "FLOPs": convert_value(model_flops),
                "Number of Parameters": convert_value(num_params)
            },
            "Invariance Properties": {
                "Mean Invariance Error": convert_value(mean_inv_err),
                "Permutation Invariance": convert_value(perm_inv)
            },
            "Validation Performance": {
                "Test Accuracy": convert_value(test_accu),
                "Test Loss": convert_value(test_loss)
            },
            "AUC Scores": {
                label: convert_value(auc) for label, auc in zip(labels, aucs)  # Convert AUCs properly
            }
                    }
    else:
        inf_times = np.array(inf_times)
        models_flops = np.array(models_flops)
        nums_params = np.array(nums_params)
        mean_inv_errs = np.array(mean_inv_errs)
        perm_invs = np.array(perm_invs)
        test_accus = np.array([acc.cpu().numpy() for acc in test_accus])
        test_losses = np.array(test_losses)
        aucs = np.array(aucs)
        
        model_info = {
            "Model Specifications": {
                "Inference Time (ms)": {
                    "Values": [convert_value(t) for t in inf_times],
                    "Mean": f"{np.mean(inf_times):.4f}",
                    "Std": f"{np.std(inf_times):.4f}"
                },
                "FLOPs": {
                    "Values": [convert_value(f) for f in models_flops],
                    "Mean": f"{np.mean(models_flops):.4f}",
                    "Std": f"{np.std(models_flops):.4f}"
                },
                "Number of Parameters": {
                    "Values": [convert_value(p) for p in nums_params],
                    "Mean": f"{np.mean(nums_params):.4f}",
                    "Std": f"{np.std(nums_params):.4f}"
                }
            },
            "Invariance Properties": {
                "Mean Invariance Error": {
                    "Values": [convert_value(e) for e in mean_inv_errs],
                    "Mean": f"{np.mean(mean_inv_errs):.4f}",
                    "Std": f"{np.std(mean_inv_errs):.4f}"
                },
                "Permutation Invariance": {
                    "Values": [convert_value(p) for p in perm_invs],
                    "Mean": f"{np.mean(perm_invs):.4f}",
                    "Std": f"{np.std(perm_invs):.4f}"
                }
            },
            "Validation Performance": {
                "Test Accuracy": {
                    "Values": [convert_value(a) for a in test_accus],
                    "Mean": f"{np.mean(test_accus):.4f}",
                    "Std": f"{np.std(test_accus):.4f}"
                },
                "Test Loss": {
                    "Values": [convert_value(l) for l in test_losses],
                    "Mean": f"{np.mean(test_losses):.4f}",
                    "Std": f"{np.std(test_losses):.4f}"
                }
            },
            "AUC Scores": {
                label: {
                    "Values": [convert_value(auc[i]) for auc in aucss],
                    "Mean": f"{np.mean([auc[i] for auc in aucss]):.4f}",
                    "Std": f"{np.std([auc[i] for auc in aucss]):.4f}"
                }
                for i, label in enumerate(labels)
            }
                        }
    
    # Define output directory and filename
    yaml_filename = os.path.join(plotdir_model_file, f"model_info.yaml")

    # Save as YAML
    with open(yaml_filename, "w") as f:
        yaml.dump(model_info, f, default_flow_style=False)

    print(f"\n{util.tcols.OKGREEN}Model characteristics saved to: {yaml_filename}{util.tcols.ENDC}\n")

####################################################################################################################
####################################################################################################################

if __name__ == "__main__":

    main()