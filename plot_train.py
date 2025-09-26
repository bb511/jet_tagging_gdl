import matplotlib.pyplot as plt
import numpy as np
import torch
import util

def weights_plot(model: torch.nn.Module, plot_dir : str = "/plot",  name = 'weights'):
    """Plots the distribution of the weights of the model per layer
        One suppose that the model is alternating between a weight matrix and a bias vector"""
    print(util.tcols.OKBLUE + "Plotting the weights of the model..." + util.tcols.ENDC)
    
    num_layers = int(len(list(model.parameters())))#//2
    print(f"Number of layers: {num_layers}")
    
    fig, axs = plt.subplots(num_layers, 1, figsize=(20, 40))
    fig.suptitle("Weights distribution per layer")
    
    for idx, parameter in enumerate(model.parameters()):
        if len(list(parameter.shape)) == 2: # Weights matrix
            axs[idx].set_title(f"Weights in layer {idx//2 + 1}")
            axs[idx].hist(parameter.detach().cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648400")
        else: # Bias vector
            axs[idx].set_title(f"Bias in layer {idx//2 + 1}")
            axs[idx].hist(parameter.detach().cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648FFF")
    plt.savefig(f"{plot_dir}/{name}_weights.png")
    plt.close()
    print(f"Weights plot saved to {plot_dir}/{name}_weights.png")
    
def gradients_plot(model: torch.nn.Module, plot_dir : str = "/plot",  name = 'gradients'):
    """Plots the distribution of the gradients of the model per layer
        One suppose that the model is alternating between a weight matrix and a bias vector"""
    print(util.tcols.OKBLUE + "Plotting the gradients of the model..." + util.tcols.ENDC)
    
    num_layers = int(len(list(model.parameters())))#//2
    print(f"Number of layers: {num_layers}")
    
    fig, axs = plt.subplots(num_layers, 1, figsize=(20, 40))
    fig.suptitle("Gradients distribution per layer")
    
    for idx, parameter in enumerate(model.parameters()):
        if len(list(parameter.shape)) == 2: # Weights matrix
            axs[idx].set_title(f"Gradients in layer {idx//2 + 1}")
            axs[idx].hist(parameter.grad.cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648400")
        else: # Bias vector
            axs[idx].set_title(f"Gradients in layer {idx//2 + 1}")
            axs[idx].hist(parameter.grad.cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648FFF")
    plt.savefig(f"{plot_dir}/{name}_gradients.png")
    plt.close()
    print(f"Gradients plot saved to {plot_dir}/{name}_gradients.png")
    
def activations_plot(model: torch.nn.Module, plot_dir : str = "/plot",  name = 'activations'):
    """Plots the distribution of the activations of the model per layer
        One suppose that the model is alternating between a weight matrix and a bias vector"""
    print(util.tcols.OKBLUE + "Plotting the activations of the model..." + util.tcols.ENDC)
    
    num_layers = int(len(list(model.parameters())))#//2
    print(f"Number of layers: {num_layers}")
    
    fig, axs = plt.subplots(num_layers, 1, figsize=(20, 40))
    fig.suptitle("Activations distribution per layer")
    
    for idx, parameter in enumerate(model.parameters()):
        if len(list(parameter.shape)) == 2: # Weights matrix
            axs[idx].set_title(f"Activations in layer {idx//2 + 1}")
            axs[idx].hist(parameter.cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648400")
        else: # Bias vector
            axs[idx].set_title(f"Activations in layer {idx//2 + 1}")
            axs[idx].hist(parameter.cpu().numpy().flatten(), bins=100, alpha = 0.5, color="#648FFF")
    plt.savefig(f"{plot_dir}/{name}_activations.png")
    plt.close()
    print(f"Activations plot saved to {plot_dir}/{name}_activations.png")
    
def plot_loss_accu(all_train_loss, all_train_accu, all_valid_loss, all_valid_accu, all_inv_err, name, data_name, loss_inv, plotdir):
    plt.plot(all_train_loss, color="gray", label="Training Loss")
    plt.plot(all_valid_loss, color="navy", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{plotdir}/Loss_{name}_{data_name}_{loss_inv}')
    plt.close()
    plt.plot(all_train_accu, color="gray", label="Training Accuracy")
    plt.plot(all_valid_accu, color="navy", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'{plotdir}/Accuracy_{name}_{data_name}_{loss_inv}')
    plt.close()
    plt.plot(all_inv_err, color="gray", label="Invariance Relative Error")
    plt.xlabel("Epochs")
    plt.ylabel("Invariance Relative Error")
    plt.savefig(f'{plotdir}/Invariance_{name}_{data_name}_{loss_inv}')
    plt.close()
    print('Finished Training')