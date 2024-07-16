import random

random.seed(15)

import numpy as np

np.random.seed(15)
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(15)
torch.cuda.manual_seed_all(15)


from deepsets import DeepSetsInvariant





path = "/mnt/c/users/bonvi/Documents/deepset/data/jetclass/ptetaphi_ptetaphi_32/"

dataset1 = torch.Tensor(np.load(path + "concatenated_x_pt_etarel_phirel_32_0.npy")).reshape(-1, 32, 3)
#dataset2 = torch.Tensor(np.load(path + "concatenated_x_pt_etarel_phirel_32_1.npy"))
#dataset3 = torch.Tensor(np.load(path + "concatenated_x_pt_etarel_phirel_32_2.npy"))
#dataset4 = torch.Tensor(np.load(path + "concatenated_x_pt_etarel_phirel_32_3.npy"))
#dataset5 = torch.Tensor(np.load(path + "concatenated_x_pt_etarel_phirel_32_4.npy"))

y1 = torch.Tensor(np.load(path + "concatenated_y_0.npy"))
#y2 = torch.Tensor(np.load(path + "concatenated_y_1.npy"))
#y3 = torch.Tensor(np.load(path + "concatenated_y_2.npy"))
#y4 = torch.Tensor(np.load(path + "concatenated_y_3.npy"))
#y5 = torch.Tensor(np.load(path + "concatenated_y_4.npy"))

print(dataset1.shape, y1.shape)

datasets = [
    torch.utils.data.TensorDataset(dataset1, y1),
    #torch.utils.data.TensorDataset(dataset2, y2),
    #torch.utils.data.TensorDataset(dataset3, y3),
    #torch.utils.data.TensorDataset(dataset4, y4),
    #torch.utils.data.TensorDataset(dataset5, y5)
]
print("Datasets loaded")
dataloaders = [
    DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=1)
    for dataset in datasets
]
print("Dataloaders created")


def train_model(net, dataloaders, device, name, data_name, epochs, lr, batch_size, loss_inv):
    print('Beginning Training')
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=80, factor=0.1, threshold=1e-4, verbose=True)

    all_train_loss = []
    all_train_accu = []

    for epoch in range(epochs):
        print(f'Beginning of Epoch {epoch + 1}/{epochs}')
        
        for dataloader_idx, dataloader in enumerate(dataloaders):
            batch_loss_sum = 0
            batch_accu_sum = 0
            totnum_batches = 0
            net.train()
            
            for x, y in dataloader:
                x = x.to(device)
                # Assuming the last column is the label
                y_true = y.to(device)
                
                y_pred = net(x)
                
                loss = criterion(y_pred, y_true)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss_sum += loss.detach()
                totnum_batches += 1
                batch_accu_sum += torch.sum(
                y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]
            ) / len(y_true)
            
            train_loss = batch_loss_sum / totnum_batches
            train_accu = batch_accu_sum / totnum_batches
            
            print(f'[{epoch + 1}, Dataset {dataloader_idx + 1}] Trained loss: {train_loss:.5f}')
            print(f'[{epoch + 1}, Dataset {dataloader_idx + 1}] Trained accuracy: {train_accu:.5f}')
        
        all_train_loss.append(train_loss.item())
        all_train_accu.append(train_accu.item())
        
        scheduler.step(train_accu)
    torch.save(net.state_dict(), f"../models/{name}_{data_name}.pt")
    print('Finished Training')
    return all_train_loss, all_train_accu

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} as device")
    
    net = DeepSetsInvariant(nbr_const=32, 
                            nbr_feat=3,
                            phi_layers= [32,32,32],
                            rho_layers= [32],
                            activ= "relu",
                            aggreg= "mean",
                            dropout= -1,
                            output_dim= 10).to(device)
    train_model(net, dataloaders, device, "DeepSetsInvariant", "JetClass_pt_eta_rel", 100, 1e-3, 256, False)