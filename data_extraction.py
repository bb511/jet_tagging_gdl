import numpy as np
import torch
#from torch_geometric.data import Data # to create the graph data
import util
import data.get_data_hls4ml as get_data_hls4ml
import data.data_augmentation as data_augmentation

import data.standardization as standardization
import torch
from typing import Tuple
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

class DataSelection:
    def __init__(self, data_name: str, train: bool = True, jets=True,
                 dataset_size: int = None, split_ratio: float = 0.8, split_seed: int = 12, total_folds: int = 5, 
                 augment_seed: int = 15, augmentation_type: str = 'scipy', augmentation_space='so3', filter_augmented: dict = {"eta":(-3,3), "pt":(0, 1600),"etarel":(-0.5,0.5), "phirel":(-0.5,0.5)},
                 norm_outside: bool = True, norm_type: str = 'minmaxcustom', norm_coef = None):
        
        self.true_data_name = data_name
        self.train = train
        self.dataset_size = dataset_size
        self.jets=jets
        
        if "data_aug" in data_name:
            nbr_feat = data_name.split("_")[-1]
            self.feat_name = "_".join(data_name.split("_")[:-3])
            self.data_name = self.feat_name + f"_{nbr_feat}"
            self.augment = True
            if self.feat_name not in ['px_py_pz', 'px_py_pz_etarel_phirel']:
                raise ValueError(f"Invalid feature name: {self.feat_name}. Expected 'px_py_pz' or 'px_py_pz_etarel_phirel'.")
            self.augmentation = data_augmentation.DataAugmentation(self.feat_name, augmentation_type=augmentation_type, augmentation_space=augmentation_space, num_rotations=8, seed=augment_seed)
            
        else:
            self.data_name = data_name
            self.feat_name = "_".join(data_name.split("_")[:-1])
            self.augment = False
            
        #self.augment_seed = augment_seed
        #self.augmentation_type = augmentation_type
        self.filter_augmented = filter_augmented
            
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.total_folds = total_folds
        
        self.norm_outside = norm_outside
        self.norm_type = norm_type
        self.norm_coef = norm_coef
        
        if self.train:
            self.train_data = get_data_hls4ml.HLS4MLData150(root=".", data_name=self.data_name, norm="nonorm", train=True)
            if self.jets:
                self.jet_data = get_data_hls4ml.HLS4MLDataJets_old(root=".", jet_name="px_py_pz", train=True)
        else:
            self.test_data = get_data_hls4ml.HLS4MLData150(root=".", data_name=self.data_name, norm="nonorm", train=False)
            if self.jets:
                self.jet_data_test = get_data_hls4ml.HLS4MLDataJets_old(root=".", jet_name="px_py_pz", train=False)
            
    def split_train(self, x, y, jet):
        # if self.dataset_size is not None:
        #     x = x[:self.dataset_size]
        #     y = y[:self.dataset_size]
        #     jet = jet[:self.dataset_size]
        if self.split_seed != None:
            if self.jets:
                x, y, jet = shuffle(x, y, jet, random_state=self.split_seed)
            else:
                x,y = shuffle(x,y, random_state=self.split_seed)
                
        size = int(self.split_ratio * x.shape[0])
        x_only_train = x[:size]
        y_only_train = y[:size]
        x_only_val = x[size:]
        y_only_val = y[size:]
        if self.jets:
            jet_only_train = jet[:size]
            jet_only_val = jet[size:]
        
        if self.dataset_size != 0:
            x_only_train = x[:self.dataset_size]
            y_only_train = y[:self.dataset_size]
            if self.jets:
                jet_only_train = jet[:self.dataset_size]
                
        if self.jets:
            return x_only_train, y_only_train, jet_only_train, x_only_val, y_only_val, jet_only_val
        else:
            return x_only_train, y_only_train, x_only_val, y_only_val
    
    def split_kfolds(self, fold_number, x, y, jet=None):
        all_labels = np.argmax(y, axis=1)
        skf = StratifiedKFold(n_splits=self.total_folds, shuffle=True, random_state=self.split_seed)
        folds = list(skf.split(np.zeros(len(all_labels)), all_labels))
        train_ids, val_ids = folds[fold_number]
        
        train_x, train_y = x[train_ids], y[train_ids]
        val_x, val_y = x[val_ids], y[val_ids]
        
        train_jet, val_jet = jet[train_ids], jet[val_ids]
        return train_x, train_y, train_jet, val_x, val_y, val_jet
        
    def augment_data(self, x, y, jets):
        x, y, jets = self.augmentation.augment_data(x, y, jets, filters=self.filter_augmented, shuffle=True)
        return x, y, jets
    
    def get_norm_params(self, x):
        self.norm_coef = standardization.fit_standardisation(self.norm_type, x, self.feat_name)
        
    def normalize_data(self, x):
        x = standardization.apply_standardisation(x, self.norm_coef)
        return x
    
    def get_train_data(self, fold_number = None):
        x = self.train_data.x_pro
        y = self.train_data.y_pro
        if self.jets:
            jet = self.jet_data.jet   
        
        if fold_number is None:
            if self.jets:
                x_train, y_train, jet_train, x_val, y_val, jet_val = self.split_train(x, y, jet)
            else:
                x_train, y_train, x_val, y_val = self.split_train(x, y, None)
        else:
            x_train, y_train, jet_train, x_val, y_val, jet_val = self.split_kfolds(fold_number, x, y, jet)
        
        if self.augment:
            x_train, y_train, jet_train = self.augment_data(x_train, y_train, jet_train)
            
        # if self.dataset_size != 0:
        #     x_train = x_train[:self.dataset_size]
        #     y_train = y_train[:self.dataset_size]
        #     if self.jets:
        #         jet_train = jet_train[:self.dataset_size]
        
        if self.norm_coef is None:
            self.get_norm_params(x_train)
            #self.get_norm_params(x) # for testing purposes
        
        if self.norm_outside:
            x_train = self.normalize_data(x_train)
            if self.split_ratio<1:
                x_val = self.normalize_data(x_val)
            
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        if self.jets:
            jet_train = torch.tensor(jet_train, dtype=torch.float32)
            jet_val = torch.tensor(jet_val, dtype=torch.float32)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Validation data shape: {x_val.shape}")
        print(f"Validation labels shape: {y_val.shape}")
        if self.jets:
            print(f"Jet training data shape: {jet_train.shape}")
            print(f"Jet validation data shape: {jet_val.shape}")
        
        if self.jets:
            return x_train, y_train, jet_train, x_val, y_val, jet_val 
        else:
            return x_train, y_train, x_val, y_val
    
    def get_test_data(self):
        
        if self.norm_outside:
            if self.norm_coef is None:
                raise ValueError("Normalization coefficients not provided")
            else:
                x_test = self.normalize_data(self.test_data.x_pro)
        else:
            x_test = self.test_data.x_pro
        jet_test = self.jet_data_test.jet
                
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(self.test_data.y_pro, dtype=torch.float32)
        jet_test = torch.tensor(jet_test, dtype=torch.float32)
        
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Test jet shape:{jet_test.shape}")
        
        return x_test, y_test, jet_test
    
    def get_data(self):
        if self.train:
            return self.get_train_data()
        else:
            return self.get_test_data()
        
        
    

