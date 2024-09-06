import os
import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import yaml
import deepspeed
from import_data_jetclass import select_data
from import_model import get_model
import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of the model you want to train")
parser.add_argument("--config_file", type=str, help="Configuration of your model", default="config_jetclass.yml")
parser.add_argument("--data_name", type=str, default='px_py_pz_8')
parser.add_argument("--n_c", type=int, default=80, help="Number of constituents")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(net, train_data, valid_data, device, name, data_name, epochs, lr, batch_size, loss_inv, warmup_steps):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)

    total_steps = len(train_data) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    print(f'Beginning of training of model {name}')
    all_train_loss, all_train_accu, all_valid_loss, all_valid_accu, all_inv_err = [], [], [], [], []
    best_accu = 0
    epochs_no_improve = 0
    epochs_es_limit = 50

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")

    prof = deepspeed.profiling.flops_profiler.FlopsProfiler(net)

    for epoch in range(epochs):
        net.train()
        batch_loss_sum, batch_accu_sum, totnum_batches = 0, 0, 0
        print(f'Beginning of Epoch {epoch + 1}/{epochs}')

        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            #x = x[:, :args.n_c, :]
            
            if epoch == 0 and totnum_batches == 0:
                prof.start_profile()
                y_pred = net(x)
                prof.stop_profile()
                print(f"Model flops: {prof.get_total_flops() / x.shape[0]}")
                prof.end_profile()
            else:
                y_pred = net(x)

            if loss_inv == 'so3':
                loss = util.inv_err_randn(net, criterion, x, y_pred, y, 0.001, device)
            elif loss_inv == 'lorentz':
                loss = util.inv_err_lorentz(net, criterion, x, y_pred, y, 0.001, device)
            elif loss_inv == 'so3_eta_phi':
                loss = util.inv_err_rand_eta_phi(net, criterion, x, y_pred, y, 0.001, device)
            else:
                loss = criterion(y_pred, torch.argmax(y, dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss_sum += loss.detach()
            batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y.max(dim=1)[1]) / len(y)
            totnum_batches += 1

        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches
        inv_err = util.check_inv(net, x, device) if 'px_py_pz' in data_name else 100

        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        all_inv_err.append(inv_err)

        print(f'[{epoch + 1}] Train loss: {train_loss:.5f}, accuracy: {train_accu:.5f}, inv_err: {inv_err:.5f}')
        print(f'[{epoch + 1}] Learning rate: {scheduler.get_last_lr()[0]}')

        # Validation
        net.eval()
        batch_loss_sum, batch_accu_sum, totnum_batches = 0, 0, 0
        with torch.no_grad():
            for x, y in valid_data:
                x, y = x.to(device), y.to(device)
                
                #x = x[:, :args.n_c, :]
                y_pred = net(x)
                
                loss = criterion(y_pred, torch.argmax(y, dim=1).to(torch.int64))
                
                batch_accu_sum += torch.sum(y_pred.max(dim=1)[1] == y.max(dim=1)[1]) / len(y)
                batch_loss_sum += loss.detach()
                totnum_batches += 1

        valid_loss = batch_loss_sum / totnum_batches
        valid_accu = batch_accu_sum / totnum_batches
        all_valid_loss.append(valid_loss.item())
        all_valid_accu.append(valid_accu.item())

        print(f"[{epoch + 1}] Valid loss: {valid_loss:.5f}, accuracy: {valid_accu:.5f}")

        if valid_accu > best_accu:
            best_accu = valid_accu
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= epochs_es_limit:
            print(f"Early stopping triggered after {epochs_es_limit} epochs without improvement")
            break

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(all_train_loss, label="Train")
    plt.plot(all_valid_loss, label="Valid")
    plt.title("Loss")
    plt.legend()

    plt.subplot(132)
    plt.plot(all_train_accu, label="Train")
    plt.plot(all_valid_accu, label="Valid")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(133)
    plt.plot(all_inv_err)
    plt.title("Invariance Error")

    plt.tight_layout()
    plt.savefig(f'plot/training_curves_{name}_{data_name}.png')
    plt.close()

    print('Finished Training')

def main(config):
    name = args.name
    assert name == "transformer", "This training script is only for the transformer model"
    data_name = args.data_name
    print(f"Training of the model {name} with data {data_name}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    outdir = util.make_output_directory(f"train_models/trained_models_{name}_{data_name}", config["outdir"])
    util.save_config_file(config, outdir)
    config_path = os.path.join(outdir, f"config_{name}_{data_name}.yml")
    print('Config file saved')

    training_data, train_sampler, validation_data = select_data(name, data_name)
    
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=config["training_hyperparams"]["batch_size"], pin_memory = True, num_workers=16, sampler=train_sampler)#, collate_fn=custom_collate
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=config["training_hyperparams"]["batch_size"], shuffle=False, pin_memory = True, num_workers=16)
    
    x, _ = next(iter(train_loader))
    print(f"Sample batch shape: {x.shape}")
    nbr_const, nbr_feat = x.shape[1], x.shape[2]
    
    model = get_model(name, nbr_const, nbr_feat, config["model_hyperparameter"]).to(device)
    print('Model summary:', model)

    train_model(model, train_loader, val_loader, device, name, data_name, **config["training_hyperparams"], warmup_steps=1000)
    print('Model training completed')

    model_file = os.path.join(outdir, f"model_{name}_{data_name}.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)

    main(config)