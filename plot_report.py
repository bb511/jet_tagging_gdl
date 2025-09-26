import yaml
import pandas as pd
import os

import data.plot_data as plot_data
import data.data_augmentation as augmentation
import data.standardization as standardization
import data.get_data_hls4ml as get_data_hls4ml
from data_extraction import DataSelection
from pathlib import Path

data_name = 'px_py_pz_pt_eta_etarel_phi_phirel_32'

train_data = get_data_hls4ml.HLS4MLData150(root=".", data_name=data_name, norm="nonorm", train=True)
x_train = train_data.x_pro
y_train = train_data.y_pro

print("✅ Train data shape:", x_train.shape)
plot_data.normalised_data_subplot(Path("hist_px_py_pz_pt_eta_phi_etarel_phirel_32"), x_train, y_train, "nonorm", "px_py_pz_pt_eta_etarel_phi_phirel")

feat_name = "_".join(data_name.split("_")[:-1])
norm_coef = standardization.fit_standardisation('minmaxcustom', x_train, feat_name)
x_norm = standardization.apply_standardisation(x_train, norm_coef)

plot_data.normalised_data_subplot(Path("hist_px_py_pz_pt_eta_phi_etarel_phirel_32"), x_norm, y_train, "norm", "px_py_pz_pt_eta_etarel_phi_phirel")