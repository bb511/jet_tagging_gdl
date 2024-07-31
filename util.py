# Utility methods to support trivial tasks done in other files.

import os
import time

import yaml
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import matplotlib.pyplot as plt

from network.deepsets import DeepSetsEquivariant
from network.deepsets import DeepSetsInvariant

from sklearn.metrics import roc_curve, auc



def make_output_directories(locations: list | str, outdir: str) -> list:
    """Create an output directory in a list of locations."""
    if isinstance(locations, str):
        return make_output_directory(locations, outdir)

    return [make_output_directory(location, outdir) for location in locations]


def make_output_directory(location: str, outdir: str) -> str:
    """Create the output directory in a designated location."""
    outdir = os.path.join(location, outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def choose_deepsets(choice: str, model_hyperparams: dict):
    """Imports a DeepSets model."""
    deepsets = {
        "ds_invariant": lambda: DeepSetsInvariant(**model_hyperparams),
        "ds_equivariant": lambda: DeepSetsEquivariant(**model_hyperparams),
    }
    model = deepsets.get(choice, lambda: None)()
    if model is None:
        raise ValueError(
            f"{choice} is not the name of a DS model. Options: {deepsets.keys()}."
        )

    print(tcols.OKBLUE + "Network architecture:" + tcols.ENDC)
    torchinfo.summary(model)

    return model




def get_model(model_type: str, model_hyperparams: dict):
    """Get a model specified through a string in the configuration file."""
    if "ds" in model_type:
        model = choose_deepsets(model_type, model_hyperparams)
    else:
        raise ValueError("Please specify in config  which kind of ML model you want.")

    return model


def choose_loss(choice, device):
    """Get a pytorch loss object given a string specified in the configuration file."""
    losses = {
        "ce": lambda: nn.CrossEntropyLoss().to(device),
        "nll": lambda: nn.NLLLoss().to(device),
    }

    loss = losses.get(choice, lambda: None)()
    if loss is None:
        raise ValueError(f"Loss {choice} not specified. Go to util.py and add it.")

    return loss


def import_data(device: str, config: dict, train: bool):
    """Imports the jet data and casts it into a torch DataLoader object."""
    print(tcols.OKGREEN + "Importing data... " + tcols.ENDC, end="")
    jet_data = data.HLS4MLData150(
        config["root"],
        config["nconst"],
        config["feats"],
        config["norm"],
        train,
    )
    if train:
        print("training data imported!")
    else:
        config["torch_dataloader"]["shuffle"] = False
        print("validation data imported!")

    dl_args = config["torch_dataloader"]
    tensor_dataset = jet_data.get_torch_dataset()
    if device == "cpu":
        return torch.utils.data.DataLoader(tensor_dataset, **dl_args)

    return torch.utils.data.DataLoader(tensor_dataset, pin_memory=True, **dl_args, drop_last=True)


def print_data_deets(data, data_type: str):
    """Prints some key details of the data set, for sanity check at start of training."""
    try:
        x, y = next(iter(data))
    except:
        x,y,_ = next(iter(data))
    try:
        print(tcols.HEADER + f"{data_type} data details:" + tcols.ENDC)
        print(f"Dataset size: {len(data)}")
        print(f"Number of constituents: {x.size(0)}")
        print(f"Number of features: {x.size(1)}")
        #print(f"Number of features: {x.size(2)}")
        print("")
        return x.size(0), x.size(1)
    except:
        print(tcols.WARNING + "WARNING: If you are trying to train a Scalar model this error message is expected! Else there is an error in the data set!" + tcols.ENDC)
        print(tcols.HEADER + f"{data_type} data details:" + tcols.ENDC)
        print(f"Dataset size: {len(data)}")
        print(f"Number of features: {x.size(0)}")
        #print(f"Number of features: {x.size(1)}")
        print("")
        return 16,4


def save_config_file(config: dict, outdir: str):
    """Saves the config file into given output directory."""
    outfile = os.path.join(outdir, "config.yml")
    with open(outfile, "w") as file:
        yaml.dump(config, file)


def load_config_file(config_file: str):
    """Loads a config file given a certain path."""
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def loss_plot(all_train_losses: list, all_valid_losses: list, outdir: str, label=""):
    """Plots the loss for each epoch for the training and validation data."""
    epochs = list(range(len(all_train_losses)))
    plt.plot(
        epochs,
        all_train_losses,
        color="gray",
        label=f"Training loss",
    )
    plt.plot(epochs, all_valid_losses, color="navy", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel(f"{label} loss")

    best_valid_loss = min(all_valid_losses)
    plt.text(
        np.min(epochs),
        np.max(all_train_losses),
        f"Min: {best_valid_loss:.3f}",
        verticalalignment="top",
        horizontalalignment="left",
        color="blue",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{label}_loss_epochs.pdf"))
    plt.close()
    print(tcols.OKGREEN + f"Loss vs epochs plot saved to {outdir}." + tcols.ENDC)


def accu_plot(all_train_accs: list, all_valid_accs: list, outdir: str):
    """Plots the loss for each epoch for the training and validation data."""
    epochs = list(range(len(all_train_accs)))
    plt.plot(
        epochs,
        all_train_accs,
        color="gray",
        label="Training Accuracy (average)",
    )
    plt.plot(epochs, all_valid_accs, color="navy", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    best_valid_acc = max(all_valid_accs)
    plt.text(
        np.min(epochs),
        np.max(all_train_accs),
        f"Min: {best_valid_acc:.3f}",
        verticalalignment="top",
        horizontalalignment="left",
        color="blue",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )
    plt.legend()
    plt.savefig(os.path.join(outdir, "accu_epochs.pdf"))
    plt.close()
    print(tcols.OKGREEN + f"Accuracy vs epochs plot saved to {outdir}." + tcols.ENDC)


class tcols:
    """Pretty terminal colors ooooo."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
import matplotlib.pyplot as plt
from sklearn import metrics

def find_nearest(array: np.ndarray, value: float):
    """Finds the index of the nearest value in an array to a given value."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def flatten_vect(x:np.array):
    """Function to flatten the input of the network"""
    x_flat = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))
    for i in range(x.shape[0]):
        x_flat[i] = x[i].flatten('C')
    return x_flat

def weights_plot(model, name = 'weights'):
    """Plots the distribution of the weights of the model per layer"""
    print(tcols.OKBLUE + "Plotting the weights of the model..." + tcols.ENDC)
    num_layers = int(len(list(model.parameters())))#//2
    print(f"Number of layers: {num_layers}")
    fig, axs = plt.subplots(num_layers, 1, figsize=(20, 40))
    fig.suptitle("Weights distribution per layer")
    for idx, parameter in enumerate(model.parameters()):
        if len(list(parameter.shape)) == 2:
            axs[idx].set_title(f"Weights in layer {idx//2 + 1}")
            axs[idx].hist(parameter.detach().cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648400")
        else:
            axs[idx].set_title(f"Bias in layer {idx//2 + 1}")
            axs[idx].hist(parameter.detach().cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648FFF")
    plt.savefig(f"plot/{name}_weights.png")
    plt.close()
    print(f"Weights plot saved to plot/{name}_weights.png")


def compute_roc_stats(all_fold_predictions, n_classes=5):
    n_folds = len(all_fold_predictions)
    tpr_base = np.linspace(0, 1, 100)
    
    all_fprs = []
    all_aucs = []
    all_fats = []
    
    for class_idx in range(n_classes):
        class_fprs = []
        class_aucs = []
        class_fats = []
        
        for fold_preds in all_fold_predictions:
            y_true = fold_preds[1][:, class_idx]
            y_pred = fold_preds[0][:, class_idx]
            
            
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            interp_fpr = np.interp(tpr_base, tpr, fpr)
            class_fprs.append(interp_fpr)
            
            roc_auc = auc(fpr, tpr)
            class_aucs.append(roc_auc)
            
            # Compute FAT (False Alarm at 80% True Positive Rate)
            idx_fat = np.argmin(np.abs(tpr - 0.8))
            fat = fpr[idx_fat]
            class_fats.append(fat)
        
        mean_fpr = np.mean(class_fprs, axis=0)
        std_fpr = np.std(class_fprs, axis=0)
        
        mean_auc = np.mean(class_aucs, axis=0)
        std_auc = np.std(class_aucs, axis=0)
        
        mean_fat = np.mean(class_fats, axis=0)
        std_fat = np.std(class_fats, axis=0)
        all_fprs.append((mean_fpr, std_fpr))
        all_aucs.append((mean_auc, std_auc))
        all_fats.append((mean_fat, std_fat))
    
    return tpr_base, all_fprs, all_aucs, all_fats

def roc_curves_uncertainty(
    all_fold_predictions,
    name,
    outdir="plot",
):
    """Plots ROC curves given fprs and tprs for each class."""

    tpr, fprs, aucs, fats = compute_roc_stats(all_fold_predictions)
    
    fprs_mean = np.array([fpr[0] for fpr in fprs])
    fprs_errs = np.array([fpr[1] for fpr in fprs])
    aucs_mean = np.array([auc[0] for auc in aucs])
    aucs_errs = np.array([auc[1] for auc in aucs])
    fats_mean = np.array([fat[0] for fat in fats])
    fats_errs = np.array([fat[1] for fat in fats])

    labels = ["Gluon", "Quark", "W", "Z", "Top"]
    cols = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    #tpr_baseline = np.linspace(0.025, 0.99, 100)
    for idx, label in enumerate(labels):
        plt.plot(
            tpr,
            fprs_mean[idx],
            color=cols[idx],
            label=f"{label}: {aucs_mean[idx]*100:.1f}% $\\pm$ {aucs_errs[idx]:.4f}; FAT: {fats_mean[idx]:.4f} $\\pm$ {fats_errs[idx]:.4f}",
        )
        plt.fill_between(
            tpr,
            fprs_mean[idx] - fprs_errs[idx],
            fprs_mean[idx] + fprs_errs[idx],
            color=cols[idx],
            alpha=0.2
        )

    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.ylim(0.001, 1)
    plt.semilogy()

    plt.legend()
    plt.savefig(os.path.join(outdir, f"roc_curves_uncertainty_{name}.pdf"))
    plt.close()
    print(f"ROC curves plot saved to {outdir}.")



def roc_curves(outdir: str, y_pred: np.ndarray, y_test: np.ndarray, name = 'roc_curves_test'):
    """Plot the ROC curves for the labels of the jet data set."""
    labels = ["Gluon", "Quark", "W", "Z", "Top"]
    cols = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    tpr_baseline = np.linspace(0.025, 0.99, 100)
    fprs = []
    aucs = []
    fprs_at_tprs = []
    for idx, label in enumerate(labels):
        fpr, tpr, thr = metrics.roc_curve(y_test[:, idx], y_pred[:, idx])
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        fpr_baseline = np.interp(tpr_baseline, tpr, fpr)
        fprs.append(fpr_baseline)
        #fpr_baseline.astype("float32").tofile(os.path.join(outdir, f"fpr_{label}_{name}.dat"))
        #tpr_baseline.astype("float32").tofile(os.path.join(outdir, f"tpr_{label}_{name}.dat"))
        tpr_idx = find_nearest(tpr, 0.8)
        plt.plot(
            tpr,
            fpr,
            color=cols[idx],
            label=f"{label}: AUC = {auc*100:.1f}%; FPR @ 80% TPR: {fpr[tpr_idx]:.3f}",
        )
        fprs_at_tprs.append(fpr[tpr_idx])

    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.ylim(0.001, 1)
    plt.semilogy()

    plt.legend()
    plt.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close()
    print(f"ROC curves plot saved to {outdir}.")

    return fprs, tpr_baseline, aucs, fprs_at_tprs

def normalise(y_pred):
    """Function to Normalize the distribution of the output of the neural network """
    print(y_pred.shape)
    y_tot = np.sum(y_pred)
    labels = ["Gluon", "Quark", "W", "Z", "Top"]
    for idx, _ in enumerate(labels):
        y_pred[:,idx] = y_pred[:,idx]/y_tot 
        
    return y_pred

def dnn_output(outdir: str, y_pred: np.ndarray, name = 'fc'):
    """Plots the output of the last part (fc) of the interaction network."""
    labels = ["Gluon", "Quark", "W", "Z", "Top"]
    cols = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    bins = np.linspace(0.0, 1.0, 20)
    #y_pred = normalise(y_pred)
    for idx, label in enumerate(labels):
        plt.hist(y_pred[:, idx], bins, label=label, histtype="step", color=cols[idx])

    plt.semilogy()
    plt.xlabel(f"${name}(x)$ DNN Output")
    plt.legend()

    plt.savefig(os.path.join(outdir, f"{name}_output_histos.pdf"))
    plt.close()
    print(f"DNN output histograms plot saved to {outdir}.")
    
def dnn_output_uncertainty(outdir: str, all_fold_predictions: np.ndarray, name = 'fc'):
    """Plots the output of the last part (fc) of the interaction network."""
    labels = ["Gluon", "Quark", "W", "Z", "Top"]
    cols = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    bins = np.linspace(0.0, 1.0, 20)
    #y_pred = normalise(y_pred)
    for idx, label in enumerate(labels):
        for fold_preds in all_fold_predictions:
            #y_true = fold_preds[1][:, idx]
            y_pred = fold_preds[0][:, idx]
            plt.hist(y_pred, bins, label=label, histtype="step", color=cols[idx])
            #plt.hist(y_true, bins, label=label, histtype="step", color=cols[idx], alpha=0.2)
            break

    plt.semilogy()
    plt.xlabel(f"${name}(x)$ DNN Output")
    plt.legend()

    plt.savefig(os.path.join(outdir, f"{name}_output_histos.pdf"))
    plt.close()
    print(f"DNN output histograms plot saved to {outdir}.")
    
def compute_inference_time(model, x, name, device, config_path):
    """Function that compute the inference time of a model"""
    batch_size = x.size(0)
    if name in ["deepset", "vector_neuron", "vector_neuron_lorentz","vn_etaphi_rel", "transformer"]:
        print('')
    else:
        raise ValueError(
            f"The model {name} is not implemented!"
            )
    data = x.to(device)
    t0 = time.time()
    y = model(data.to(device))
    t1 = time.time()
    inf_time = (t1-t0)/ batch_size
    print(f"The time take for inference over {1} datapoints, was {inf_time:.2E} seconds")


    
def check_invariance(model, x, name, device, config_path, val = False):
    nbr_const, nbr_feat = x.size(1), x.size(2)
    try:
        assert nbr_feat == 3, "The number of features should be 3"
    except:
        if nbr_feat == 5:
            print(tcols.WARNING + "The number of features is 5, this should be the case only if you work with px,py,pz, eta_rel, phi_rel" + tcols.ENDC)
        
        else:
            print("The number of features should be 3 for the invariance check!")
            return 1000000.0

    batch_size = 1000
    if name in ["deepset", "vector_neuron", "vector_neuron_lorentz", "vn_etaphi_rel", "transformer"]:
        input = x.to(device)
        rot_mats = []
        for _ in range(16):
            R = np.random.rand(3, 3)
            U, _, __ = np.linalg.svd(R)
            #generate an element of SO(3)
            R = U@U
            rot_mats.append(R)
        err = np.zeros(len(rot_mats))
        for i in range(len(rot_mats)):
            a = model(input)
            
            if nbr_feat == 5:
                rot_input = input[:, :, :3] @ torch.tensor(rot_mats[i]).to(torch.float32).to(device)
                concat_input = torch.cat((rot_input, input[:,:,3:]), dim = 2)
                b = model(concat_input)
            else:
                b = model(input @ torch.tensor(rot_mats[i]).to(torch.float32).to(device))
            
            err[i] = torch.norm(b-a)/torch.norm(a)
        mean_err = err.mean()
        print(f'The mean invariance error is {mean_err*100}%')
    else:
        raise ValueError(
            f"The model {name} is not implemented!"
            )
        
def check_invariance_Lorentz(model, x, name, device, val = False):
    """Function that checks the invariance of the model to Lorentz transformations"""
    nbr_const, nbr_feat = x.shape[1], x.shape[2]
    try:
        assert nbr_feat == 4, "The number of features should be 4"
    except:
        if nbr_feat == 5:
            print(tcols.WARNING + "The number of features is 5, this should be the case only if you work with px,py,pz, eta_rel, phi_rel" + tcols.ENDC)
        
        else:
            print("The number of features should be 4 for the invariance check!")
            return 1000000.0

    batch_size = 1000
    if name in ["deepset", "vector_neuron", "vector_neuron_lorentz","vn_etaphi_rel", "transformer"]:
        input = x.to(device)
        rot_mats = []
        for _ in range(16):
            R = np.random.rand(3, 3)
            U, _, V = np.linalg.svd(R)
            #generate two elements of SO(3)
            P = U@U
            Q = V@V
            #R = U
            M = np.zeros((4,4))
            M[:-1, :-1] = P
            M[-1,-1]= 1
            M = torch.tensor(M).to(torch.float32).to(device)
            N = np.zeros((4,4))
            N[:-1, :-1] = np.transpose(Q)
            N[-1,-1]= 1
            N = torch.tensor(N).to(torch.float32).to(device)
            rot_mats.append((M,N))
        rapidity = torch.rand(len(rot_mats))
        err = torch.zeros(len(rot_mats))
        for i in range(len(rot_mats)):
            # any matrix in SO(3,1) can be written as a product of the following matrices
            matrix = rot_mats[i][0] @ torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],[0,0,torch.cosh(rapidity[i]), torch.sinh(rapidity[i])], [0,0,torch.sinh(rapidity[i]), torch.cosh(rapidity[i])]]).to(torch.float32).to(device) @ rot_mats[i][1]
            matrix = torch.transpose(matrix,1,0)
            a = model(input)
            
            if nbr_feat == 5:
                rot_input = input[:, :, :3] @ matrix
                concat_input = torch.cat((rot_input, input[:,:,3:]), dim = 2)
                b = model(concat_input)
            else:
                
                b = model(input @ matrix)
            
            err[i] = torch.norm(b-a)/torch.norm(a)
        mean_err = err.mean()
        print(f'The mean invariance error is {mean_err*100}% for Lorentz transformations')
    else:
        raise ValueError(
            f"The model {name} is not implemented!"
            )
        

def create_rot_mat(theta, phi, psi, device):
    """Function to create a rotation matrices from the Euler angles"""
    dic_rot = {}
    Rz = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]]).to(device)
    Ry = torch.tensor([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]]).to(device)
    Rx = torch.tensor([[1, 0, 0],
                    [0, np.cos(psi), -np.sin(psi)],
                    [0, np.sin(psi), np.cos(psi)]]).to(device)
    dic_rot['R'] = Rz @ Ry @ Rx
    angles = np.linspace(0, 2*np.pi, 8)
    for angle_theta in angles:
        for angle_phi in angles:
            for angle_psi in angles:
                Rz = torch.tensor([[np.cos(theta + angle_theta), -np.sin(theta + angle_theta), 0],
                            [np.sin(theta + angle_theta), np.cos(theta + angle_theta), 0],
                            [0, 0, 1]])
                Ry =torch.tensor([[np.cos(phi + angle_phi), 0, np.sin(phi + angle_phi)],
                            [0, 1, 0],
                            [-np.sin(phi + angle_phi), 0, np.cos(phi + angle_phi)]])
                Rx = torch.tensor([[1, 0, 0],
                            [0, np.cos(psi + angle_psi), -np.sin(psi + angle_psi)],
                            [0, np.sin(psi + angle_psi), np.cos(psi + angle_psi)]])
                dic_rot[f'{angle_theta}, {angle_phi}, {angle_psi}'] = Rz @ Ry @ Rx
    dic_rot['Id'] = torch.eye(3)
    return dic_rot


def inv_err(model, criterion, x, y_pred, y_true, l, device):
    """
    Calculates the invariant error for a given model and input data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        x (torch.Tensor): The input data.
        y_pred (torch.Tensor): The predicted output.
        y_true (torch.Tensor): The true output.
        l (float): The weight for the mean error term.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The total loss, including the mean error and the invariant loss.
    """
    batch_size = 256
    input = torch.randn((batch_size, x.shape[1], x.shape[2])).to(device)
    theta, phi, psi = np.random.rand(3)
    rot_mats = create_rot_mat(theta, phi, psi, device)
    err = torch.zeros(rot_mats.shape[0])
    for i in rot_mats.keys():
        #input = from_pt_eta_phi_to_px_py_pz(input)
        a = model(input)
        b = model(input @ rot_mats[i])
        err[i] = torch.norm(b-a)#/torch.norm(a)
    mean_err = err.mean()
    #print(f'The mean invariance error is {mean_err*100}%')
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_err_rand(model, criterion, x, y_pred, y_true, l, device):
    input = x
    rot_mats = torch.zeros((16, 3,3)).to(device)
    for i in range(rot_mats.shape[0]):
        R = np.random.rand(3, 3)
        U, _, __ = np.linalg.svd(R)
        #generate an element of SO(3)
        R = U@U
        rot_mats[i] = torch.tensor(R).to(torch.float32).to(device)
    err = torch.zeros(rot_mats.shape[0])
    for i in range(rot_mats.shape[0]):
        a = model(input)
        b = model(input @ rot_mats[i])
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_err_rand_eta_phi(model, criterion, x, y_pred, y_true, l, device):
    input = x
    assert x.shape[2] == 5, "The number of features should be 5"
    rot_mats = torch.zeros((16, 3,3)).to(device)
    for i in range(rot_mats.shape[0]):
        R = np.random.rand(3, 3)
        U, _, __ = np.linalg.svd(R)
        #generate an element of SO(3)
        R = U@U
        rot_mats[i] = torch.tensor(R).to(torch.float32).to(device)
    err = torch.zeros(rot_mats.shape[0])
    for i in range(rot_mats.shape[0]):
        a = model(input)
        px_py_pz = input[:, :, :3]
        rot_input = px_py_pz @ rot_mats[i]
        concat_input = torch.cat((rot_input, input[:,:,3:]), dim = 2)
        b = model(concat_input)
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_error_lorentz(model, criterion, x, y_pred, y_true, l, device):
    """Function that checks the invariance of the model to Lorentz transformations"""
    
    input = x
    rot_mats = []
    for _ in range(16):
        R = np.random.rand(3, 3)
        U, _, V = np.linalg.svd(R)
        #generate two elements of SO(3)
        P = U@U
        Q = V@V
        #R = U
        M = np.zeros((4,4))
        M[:-1, :-1] = P
        M[-1,-1]= 1
        M = torch.tensor(M).to(torch.float32).to(device)
        N = np.zeros((4,4))
        N[:-1, :-1] = np.transpose(Q)
        N[-1,-1]= 1
        N = torch.tensor(N).to(torch.float32).to(device)
        rot_mats.append((M,N))
    rapidity = torch.rand(len(rot_mats))
    err = torch.zeros(len(rot_mats))
    for i in range(len(rot_mats)):
        # any matrix in SO(3,1) can be written as a product of the following matrices
        matrix = rot_mats[i][0] @ torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],[0,0,torch.cosh(rapidity[i]), torch.sinh(rapidity[i])], [0,0,torch.sinh(rapidity[i]), torch.cosh(rapidity[i])]]).to(torch.float32).to(device) @ rot_mats[i][1]
        matrix = torch.transpose(matrix,1,0)
        a = model(input)
        b = model(input @ matrix)
        
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def permute_constituents(data: torch.Tensor):
    """Applies a random translation to each data point."""
    permutation = torch.randperm(data.size(1))
    data = data[:, permutation]

    return data

def check_permutation_invariance(model, x, name, device, config_path, val = False):
    """Function that checks the permutation invariance of the model"""
    batch_size = 1000  
    input = x.to(device)
    err = np.zeros(16)
    for i in range(err.shape[0]):
        a = model(input)
        b = model(permute_constituents(input).to(device))
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    print("The mean permutation invariance error is ", mean_err * 100, "%")
    return mean_err

def permutation_err(model, criterion, x, y_pred, y_true, l, device):
    """
    Calculates the invariant error for a given model and input data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        x (torch.Tensor): The input data.
        y_pred (torch.Tensor): The predicted output.
        y_true (torch.Tensor): The true output.
        l (float): The weight for the mean error term.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The total loss, including the mean error and the invariant loss.
    """
    batch_size = 256
    input = torch.randn((batch_size, x.shape[1], x.shape[2])).to(device)
    err = np.zeros(16)
    for i in range(err.shape[0]):
        a = model(input)
        b = model(permute_constituents(input).to(device))
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def check_inv(model, x, device):
    model.eval()
    input = x
    rot_mats = []
    for _ in range(16):
        R = np.random.rand(3, 3)
        U, _, __ = np.linalg.svd(R)
        #generate an element of SO(3)
        R = U@U
        rot_mats.append(R)
    err = np.zeros(len(rot_mats))
    for i in range(len(rot_mats)):
        input = from_pt_eta_phi_to_px_py_pz(input)
        a = model(input)
        b = model(input @ torch.tensor(rot_mats[i]).to(torch.float32).to(device))
        err[i] = torch.norm(b-a)/torch.norm(a)
            #print(torch.norm(b-a), torch.norm(a))
            #c = model(input) @ rot_mat
    mean_err = err.mean()
    #print(f'The mean invariance error is {mean_err*100}%')
    return mean_err

def augerino_loss(model, criterion, y_pred, y_true):
    theta = model.theta_tilde
    return  criterion(y_pred, torch.argmax(y_true, dim = 1)) - 0.1 * torch.norm(theta)**2

def from_pt_eta_phi_to_px_py_pz(data):
    """Function that transform pt, eta, phi to px, py, pz"""
    pt = data[:,:,0]
    eta = data[:,:,1]
    phi = data[:,:,2]
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return torch.stack((px, py, pz), dim = 2)

def from_pt_eta_phi_to_px_py_pz_np_array(data):
    """Function that transform pt, eta, phi to px, py, pz"""
    pt = data[:,:,0]
    eta = data[:,:,1]
    phi = data[:,:,2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack((px, py, pz), axis = 2)


def preprocess_scalar(x: np.array) -> np.array:
    """function to generate the inputs in order to make the network invariant to SO(3)
        Following: https://arxiv.org/pdf/2106.06610.pdf Lemma 2
        The scalar product of each pairs of inputs as well as the determinant of all the  3*3 submatrices  is 
        computed.
        TO DO: make this more efficient"""
    # initialize the vector containing the scalar products. It has size (batch, nbr_const**2)
    z = np.zeros((x.shape[0],x.shape[1]*x.shape[1]))
    
    # compute the scalar product of each vector
    for i in range(x.shape[0]):
        z[i,:] = (x[i,:,:] @ np.transpose(x[i,:,:])).flatten('C') 
    
    # initialize the vector that will contain the 3*3 subdeterminants. It has shape (batch, nbr_const*(nbr_const-1)*(nbr_const-2)/6)
    # There are nbr_const to choose the first vector, nbr_const - 1 the second and nbr_const - 2 to choose the third
    # we divide by the number of same determinant we compute (# ways to put 3 different balls in 3 boxes = 6)
    det = np.zeros((x.shape[0], int(x.shape[1] * (x.shape[1] - 1) * (x.shape[1] - 2) / 6)))
    
    # compute the determinant
    for m in range(x.shape[0]):
        count = 0
        for i in range (x.shape[1]):
            for j in range(x.shape[1]):
                for k in range(x.shape[1]):
                    # choose an ordening such that we do not get multiple times the determinant of the same 3*3 submatrix
                    if i < j and j < k:
                        det[m,count] = np.linalg.det(np.concatenate((np.expand_dims(x[m,i], -1), np.expand_dims(x[m,j], -1), np.expand_dims(x[m,k], -1)), axis = 1))
                        count += 1
    
    # concatenate the result of the scalar products and the subdeterminants computations
    z = np.concatenate((z,det),axis=1)
    
    return z

def inv_err_phi(model, criterion, x, y_pred, y_true, l, device):
    """
    Calculates the invariant error for a given model and input data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        x (torch.Tensor): The input data.
        y_pred (torch.Tensor): The predicted output.
        y_true (torch.Tensor): The true output.
        l (float): The weight for the mean error term.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The total loss, including the mean error and the invariant loss.
    """
    batch_size = 256
    input = torch.randn((batch_size, x.shape[1], x.shape[2])).to(device)
    phi = np.random.rand(1)
    phi_list = np.linspace(0, 2*np.pi, 36)
    err = torch.zeros(phi_list.shape[0])
    for i in range(phi_list.shape[0]):
        a = model(input)
        phi_dim = input[:, :, 2] + phi_list[i]
        rot_input = torch.cat((input[:, :, 0:2], phi_dim.unsqueeze(-1)), dim = 2)
        b = model(rot_input)
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))
    
def no_zero_loss(net, criterion, x, y_pred, y_true):
    batch_size = 256
    input = torch.randn((batch_size, 16, x.shape[2]))
    rot_mats = torch.zeros((16,3,3))
    for i in range(rot_mats.shape[0]):
        R = np.random.rand(3, 3)
        U, _, __ = np.linalg.svd(R)
        #generate an element of SO(3)
        R = U@U
        rot_mats[i] = torch.tensor(R).to(torch.float32)
    err = torch.zeros(rot_mats.shape[0])
    for i in range(rot_mats.shape[0]): 
        a = net.pre_mlp(input).reshape(input.shape[0], input.shape[1], input.shape[2])@ rot_mats[i]
        b = net.pre_mlp(input @ rot_mats[i]).reshape(input.shape[0], input.shape[1], input.shape[2])
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.detach().mean()
    return 0.01 * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif isinstance(m, nn.MultiheadAttention):
        # Initialize the weights for all linear transformations in MultiheadAttention
        for param in m.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    elif isinstance(m, nn.TransformerEncoderLayer):
        # Initialize the feedforward layer
        nn.init.xavier_uniform_(m.linear1.weight)
        nn.init.zeros_(m.linear1.bias)
        nn.init.xavier_uniform_(m.linear2.weight)
        nn.init.zeros_(m.linear2.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def init_transformer_weights(model):
    """Initialize weights for Transformer models."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Special initialization for the final layer
    if hasattr(model, 'fc'):
        nn.init.normal_(model.fc.weight, mean=0, std=0.02)
        nn.init.zeros_(model.fc.bias)

    # Optionally, adjust the initialization of positional embeddings
    if hasattr(model, 'pos_encoder') and hasattr(model.pos_encoder, 'pe'):
        with torch.no_grad():
            model.pos_encoder.pe.div_(math.sqrt(model.d_model))