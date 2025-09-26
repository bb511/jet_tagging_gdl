import os
import torch 
import torch.optim as optim

import json
import math

import custom_loss as custom_loss
import util
import plot_train

from torch.utils.data import DataLoader, Subset

######################################################################################################################################
     
def train_model(net, train_data, valid_data, device, criterion, optimizer, scheduler,
                name, data_name, loss_inv, rot_type, rot_space, filters, compare_to_label,
                outdir, file_name, plotdir, coef_norm,
                epochs, batch_size, lr, l = 0.001):
    "Standard training function for pytorch model"
    print(util.tcols.OKCYAN + f"Training of the model {name} with data {data_name} and using loss {loss_inv} with the following hyperparameters: epochs = {epochs}, batch_size = {batch_size}, learning rate = {lr}" + util.tcols.ENDC + '\n')
    
    print(f'Begining of training of model {name}')
    all_train_loss = []
    all_train_accu = []
    all_valid_loss = []
    all_valid_accu = []
    all_valid_inv_err = []
    all_lr = []
    
    epochs_no_improve = 0
    epochs_es_limit = 50
    best_accu = 0
    
    # Evaluate invariance error on a subset of 100 samples to save time
    val_subset = Subset(valid_data.dataset, range(100))
    val_subset_loader = DataLoader(val_subset, batch_size=100, shuffle=False)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        print(f'Begining of Epoch {epoch + 1}/{epochs}')
        print("Gradient is enabled: ", torch.is_grad_enabled())
        print("Model is in train mode: ", net.training)
        for x, y_true, jet in train_data:
            
            x, y_true, jet = x.to(device), y_true.to(device), jet.to(device)
            y_pred = net(x)
            
            # compute the loss function based on the user choice
            # elif loss_inv == 'lorentz':
            #     loss = custom_loss.inv_err_lorentz(net, criterion, x, y_pred, y_true, l, 16, device)
            if loss_inv == 'so3_eta_phi' or loss_inv == 'so3':
                loss = custom_loss.inv_err_so3(net, criterion, x, y_pred, y_true, jet, coef_norm, l, rot_type, 16, device, filters, compare_to_label)
            elif loss_inv == 'so2_eta_phi' or loss_inv == 'so2':
                loss = custom_loss.inv_err_so2(net, criterion, x, y_pred, y_true, coef_norm, l, 16, device, compare_to_label)
            else:
                loss = criterion(y_pred, torch.argmax(y_true, dim = 1))
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if name == "transformer":
                scheduler.step()
            
            batch_loss_sum += loss.detach()
            totnum_batches += 1
            batch_accu_sum += torch.sum(
                y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]
            ) / len(y_true)
            
        # print statistics
        train_loss = batch_loss_sum / totnum_batches
        train_accu = batch_accu_sum / totnum_batches
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        lr_after_epoch = optimizer.state_dict()['param_groups'][0]['lr']
        all_lr.append(lr_after_epoch)
        
        print(f'[{epoch + 1}] Trained loss: {train_loss:.5f}')
        print(f'[{epoch + 1}] Trained accuracy: {train_accu:.5f}')
        print(f'[{epoch + 1}] Learning rate: ', lr_after_epoch)
        

        batch_loss_sum = 0
        batch_accu_sum = 0
        totnum_batches = 0
        #inv_err_sum = 0
        
        net.eval()
        with torch.no_grad():
            print("Gradient is enabled: ", torch.is_grad_enabled())
            print("Model is in train mode: ", net.training)
            
            for x,y, jet in valid_data:
                x = x.to(device)
                y = y.to(device)       
                jet = jet.to(device) 

                y_pred = net(x)
                y_true = y
                
                #loss = criterion(y_pred, torch.argmax(y_true, dim = 1).to(torch.int64))
                if loss_inv == 'so3_eta_phi' or loss_inv == 'so3':
                    loss = custom_loss.inv_err_so3(net, criterion, x, y_pred, y_true, jet, coef_norm, l, rot_type, 16, device, filters)
                elif loss_inv == 'so2_eta_phi'or loss_inv == 'so2':
                    loss = custom_loss.inv_err_so2(net, criterion, x, y_pred, y_true, coef_norm, l, 16, device)
                else:
                    loss = criterion(y_pred, torch.argmax(y_true, dim = 1))
                
                batch_accu_sum += torch.sum(
                    y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]
                    ) / len(y_true)
                batch_loss_sum += loss.detach()
                totnum_batches += 1 
                
            if "px_py_pz" in data_name:   
                for x, y, jet in val_subset_loader:
                    x = x.to(device)
                    y = y.to(device)
                    jet = jet.to(device)
                    
                    # y_pred = net(x)
                    # y_true = y
                    
                    # if rot_space == "so3":
                    #     inv_err = custom_loss.inv_err_so3(net, criterion, x, y_pred, y_true, jet, coef_norm, l, rot_type, 16, device, filters, return_only_error=True)
                    # elif rot_space == "so2":
                    #     inv_err = custom_loss.inv_err_so2(net, criterion, x, y_pred, y_true, coef_norm, l, 16, device, return_only_error=True)
                    if rot_space == "so3":
                        inv_err = custom_loss.check_so3_rot_err(net, x, jet, coef_norm, rot_type, 16, device, filters)
                    elif rot_space == "so2":
                        inv_err = custom_loss.check_so2_rot_err(net, x, coef_norm, 16, device)
            else:
                inv_err = 100
                
            #inv_err_sum += inv_err
                
        valid_loss = batch_loss_sum / totnum_batches
        valid_accu = batch_accu_sum / totnum_batches
        # inv_err = inv_err_sum / totnum_batches
            
        print(f"[{epoch + 1}] validation loss: {valid_loss:.5f}")
        print(f"[{epoch + 1}] Validation accuracy: {valid_accu:.5f}")
        print(f'[{epoch + 1}] Invariance relative error: {inv_err:.5f}')
        
        all_valid_loss.append(valid_loss.item())
        all_valid_accu.append(valid_accu.item())
        all_valid_inv_err.append(inv_err)
            
        model_file = os.path.join(outdir, f"{file_name}.pt")
        torch.save(net.state_dict(), model_file)
            
        if valid_accu > best_accu:
            best_model_file = os.path.join(outdir, f"{file_name}_best_val.pt")
            torch.save(net.state_dict(), best_model_file)
            print(f'Best model saved at {best_model_file}')
                
        if round(valid_accu.cpu().item(), 3) <= round(best_accu, 3):
            epochs_no_improve += 1
        else:
            best_accu = valid_accu.cpu().item()
            epochs_no_improve = 0
        
        if name != "transformer":
            scheduler.step(best_accu)
        
        # DISABLED EARLY STOPPING FOR THE MOMENT
        # if epochs_no_improve >= epochs_es_limit:
        #     if name in ["vector_neuron", "vector_neuron_so2"]:
        #         continue
        #     else:
        #         break
        
        # Save the loss values at each epoch for plot reproductibility
        with open(f"{outdir}/loss.json", "w") as f:
            json.dump(all_train_loss, f)
        with open(f"{outdir}/accu.json", "w") as f:
            json.dump(all_train_accu, f)
        with open(f"{outdir}/val_loss.json", "w") as f:
            json.dump(all_valid_loss, f)
        with open(f"{outdir}/val_accu.json", "w") as f:
            json.dump(all_valid_accu, f)
        with open(f"{outdir}/val_inv_err.json", "w") as f:
            json.dump(all_valid_inv_err, f)
        with open(f"{outdir}/lr.json", "w") as f:
            json.dump(all_lr, f)
        
        plot_train.plot_loss_accu(all_train_loss, all_train_accu, all_valid_loss, all_valid_accu, all_valid_inv_err, name, data_name, loss_inv, plotdir)
    
######################################################################################################################################    
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """Custom learning rate scheduler for transformers"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)



