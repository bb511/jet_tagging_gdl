import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Data
nbr_const_ds = [1,4,8,12,16,20,24,28,32]
accu_deepset = [0.46, 0.58, 0.63, 0.67, 0.72, 0.74, 0.75, 0.77, 0.77]
accu_deepset_weak_sym = [0.47, 0.58, 0.65, 0.67, 0.72, 0.74, 0.75, 0.76, 0.79]
nbr_const_vn = [2, 5, 9, 13, 17, 21, 25, 29, 33]
accu_vn = [0.48, 0.61, 0.69, 0.71, 0.72, 0.73, 0.73, 0.74, 0.74]
nbr_const_trans = [1,4,8,12, 16, 20, 24, 28, 32]
accu_trans = [0.46, 0.59, 0.66, 0.70, 0.73, 0.76, 0.78, 0.79, 0.79]

nbr_const_rest = [8, 16, 32]
accu_ds_aug = [0.66, 0.72, 0.79]
accu_trans_aug = [0.67, 0.74, 0.79]

# Set up the plot
plt.figure(figsize=(12, 8))
plt.style.use('default')
ax = plt.gca()
ax.set_facecolor('#f0f0f0')
for spine in ax.spines.values():
    spine.set_edgecolor('#cccccc')
    spine.set_linewidth(2)

# Color palette
colors = get_cmap('Set2')

# Create the plot
plt.plot(nbr_const_ds, accu_deepset, color=colors(0), marker='o', linewidth=1.5, markersize=8, label="Deepset")
plt.plot(nbr_const_vn, accu_vn, color=colors(1), marker='s', linewidth=1.5, markersize=8, label="Vector Neuron")
plt.plot(nbr_const_ds, accu_deepset_weak_sym, color=colors(2), marker='^', linewidth=1.5, markersize=8, label="Deepset with weak symmetry")
plt.plot(nbr_const_trans, accu_trans, color=colors(3), marker='D', linewidth=1.5, markersize=8, label="Transformer")
plt.plot(nbr_const_rest, accu_ds_aug, color=colors(4), marker='P', linewidth=1.5, markersize=8, label="Deepset with data augmentation")
plt.plot(nbr_const_rest, accu_trans_aug, color=colors(5), marker='X', linewidth=1.5, markersize=8, label="Transformer with data augmentation")

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

# Add minor ticks
plt.minorticks_on()

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()