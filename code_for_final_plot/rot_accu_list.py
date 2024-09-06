import numpy as np
import matplotlib.pyplot as plt
import tol_colors as tc

# Data (unchanged)
nbr_const_ds = [1,4,8,12,16,20,24,28,32]
accu_deepset = [0.46, 0.58, 0.63, 0.67, 0.72, 0.74, 0.75, 0.77, 0.77]
nbr_const_vn = [2, 5, 9, 13, 17, 21, 25, 29, 33]
accu_vn = [0.48, 0.61, 0.69, 0.71, 0.72, 0.73, 0.73, 0.74, 0.74]
nbr_const_trans = [1,4,8,12, 16, 20, 24, 28, 32]
accu_trans = [0.46, 0.59, 0.66, 0.70, 0.73, 0.76, 0.78, 0.79, 0.79]
nbr_const_rest = [8, 16, 32]
uncertainty_deepset = [0.001, 0.001, 0.001]
uncertainty_vn = [0.007, 0.017, 0.016]
uncertainty_trans = [0.004, 0.002, 0.001]
uncertainty_deepset_weak_sym = [0.004, 0.002, 0.001]
uncertainty_ds_aug = [0.003, 0.002, 0.001]
uncertainty_trans_weak_sym = [0, 0.001, 0.002]
uncertainty_trans_aug = [0.001, 0.002, 0.001]
accu_deepset_weak_sym = [0.65, 0.72, 0.79]
accu_ds_aug = [0.66, 0.72, 0.79]
accu_trans_aug = [0.67, 0.75, 0.80]
accu_trans_weak_sym = [0.66, 0.73, 0.77]

# Set up the plot
plt.figure(figsize=(12, 8))
plt.rc('axes', prop_cycle=plt.cycler('color', list(tc.tol_cset('bright'))))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Create the plot with error bars
plt.errorbar(nbr_const_vn, accu_vn, yerr=[0]*2 + [uncertainty_vn[0], 0, uncertainty_vn[1], 0, 0, 0, uncertainty_vn[2]], 
             fmt='s-', linewidth=1.5, markersize=8, capsize=5, label="Vector Neuron", alpha=1)
plt.errorbar(nbr_const_ds, accu_deepset, yerr=[0]*2 + [uncertainty_deepset[0], 0, uncertainty_deepset[1], 0, 0, 0, uncertainty_deepset[2]], 
             fmt='o-', linewidth=1.5, markersize=8, capsize=5, label="Deepset", alpha=1)
plt.errorbar(nbr_const_trans, accu_trans, yerr=[0]*2 + [uncertainty_trans[0], 0, uncertainty_trans[1], 0, 0, 0, uncertainty_trans[2]], 
             fmt='D-', linewidth=1.5, markersize=8, capsize=5, label="Transformer", alpha=1)
plt.errorbar(nbr_const_rest, accu_deepset_weak_sym, yerr=uncertainty_deepset_weak_sym, 
             fmt='^-', linewidth=1.5, markersize=8, capsize=5, label="Deepset with weak symmetry", alpha=1)
plt.errorbar(nbr_const_rest, accu_trans_weak_sym, yerr=uncertainty_trans_weak_sym, 
             fmt='v-', linewidth=1.5, markersize=8, capsize=5, label="Transformer with weak symmetry", alpha=1)
plt.errorbar(nbr_const_rest, accu_ds_aug, yerr=uncertainty_ds_aug, 
             fmt='P-', linewidth=1.5, markersize=8, capsize=5, label="Deepset with data augmentation", alpha=1)
plt.errorbar(nbr_const_rest, accu_trans_aug, yerr=uncertainty_trans_aug, 
             fmt='X-', linewidth=1.5, markersize=8, capsize=5, label="Transformer with data augmentation", alpha=1)

# Customize the plot
plt.xlabel("Number of constituents", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
plt.title("Model Accuracy vs Number of Constituents", fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='lower right', framealpha=0.9)

# Set axis limits
plt.xlim(0, 35)
plt.ylim(0.45, 0.85)

# Customize ticks
plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=2)
plt.tick_params(axis='both', which='minor', length=3, width=1)
plt.minorticks_on()

# Adjust layout and display
plt.tight_layout()
plt.savefig("./all_accu_with_uncertainty.png")
plt.show()