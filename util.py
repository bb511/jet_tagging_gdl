# Utility methods to support trivial tasks done in other files.
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import deepspeed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import metrics

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


def make_output_directories(locations: list | str, outdir: str) -> list:
    """Create an output directory in a list of locations."""
    if isinstance(locations, str):
        return make_output_directory(locations, outdir)

    return [make_output_directory(location, outdir) for location in locations]


def make_output_directory(location: str, outdir: str) -> str:
    """Create the output directory in a designated location."""
    outdir = os.path.join(location, outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    return outdir

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

def find_nearest(array: np.ndarray, value: float):
    """Finds the index of the nearest value in an array to a given value."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def roc_curves(outdir: str, y_pred: np.ndarray, y_test: np.ndarray, name = 'roc_curves_test'):
    """Plot the ROC curves for the labels of the jet data set."""
    labels = ["Quark", "Gluon", "W", "Z", "Top"]
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
    
def compute_inference_time(model:nn.Module, x:torch.Tensor, device):
    """Function that compute the inference time of a model"""
    with torch.no_grad():
        batch_size = x.size(0)
        data = x.to(device)
        t0 = time.time()
        y = model(data.to(device))
        t1 = time.time()
        inf_time = (t1-t0)/ batch_size
        print(f"The time take for inference over {1} datapoints, was {inf_time:.2E} seconds")
        return inf_time
    
def compute_flops(model, x, device):
    """Function that compute the number of flops of a model"""
    with torch.no_grad():
        x = x.to(device)
        prof = deepspeed.profiling.flops_profiler.FlopsProfiler(model)
        prof.start_profile()
        model(x)
        prof.stop_profile()
        print(tcols.OKGREEN + "Model flops: " + tcols.ENDC, end="")
        print(f"{prof.get_total_flops()/ x.shape[0]}")
        
        flops = prof.get_total_flops()/ x.shape[0]
        prof.end_profile()
        return flops
        
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


        

