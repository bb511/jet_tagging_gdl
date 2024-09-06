import numpy as np
import os
from tqdm import tqdm
from scipy.stats import qmc
import argparse

def create_rot_mat_sobol(num_rotations):
    sobol_engine = qmc.Sobol(d=4, scramble=True)
    sobol_samples = sobol_engine.random(n=num_rotations)
    
    rot_matrices = []
    for sample in sobol_samples:
        q = sample_to_quaternion(sample)
        rot_matrices.append(quaternion_to_rotation_matrix(q))
    
    return np.array(rot_matrices)

def create_rot_mat_random(num_rotations):
    rot_matrices = []
    for _ in range(num_rotations):
        A = np.random.randn(3, 3)
        Q, R = np.linalg.qr(A)
        rot_matrices.append(Q * np.sign(np.diag(R)))
    return np.array(rot_matrices)

def create_rot_mat_regular(num_rotations):
    rot_matrices = []
    angles = np.linspace(0, 2*np.pi, int(np.cbrt(num_rotations)), endpoint=False)
   
    for theta in angles:
        for phi in angles:
            for psi in angles:
                Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
                Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                               [0, 1, 0],
                               [-np.sin(phi), 0, np.cos(phi)]])
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(psi), -np.sin(psi)],
                               [0, np.sin(psi), np.cos(psi)]])
                rot_matrices.append(Rz @ Ry @ Rx)
   
    return np.array(rot_matrices[:num_rotations])

def sample_to_quaternion(sample):
    theta1 = 2 * np.pi * sample[0]
    theta2 = 2 * np.pi * sample[1]
    u = sample[2]
    v = sample[3]
    
    q = np.array([
        np.sqrt(1 - u) * np.sin(theta1),
        np.sqrt(1 - u) * np.cos(theta1),
        np.sqrt(u) * np.sin(theta2),
        np.sqrt(u) * np.cos(theta2)
    ])
    
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def augment_data(data, labels, rotation_matrices):
    data_px_py_pz = data[:, :, :3] 
    data_pt_eta = data[:, :, 3:]    
    
    augmented_data = np.empty((data.shape[0] * len(rotation_matrices), data.shape[1], data.shape[2]), dtype=data.dtype)
    augmented_labels = np.empty((labels.shape[0] * len(rotation_matrices), labels.shape[1]), dtype=labels.dtype)
    
    for i, rotation_matrix in enumerate(tqdm(rotation_matrices, desc="Augmenting data")):
        rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
        
        augmented_data[i::len(rotation_matrices)] = np.concatenate((data_pt_eta, rotated_data), axis=2)
        augmented_labels[i::len(rotation_matrices)] = labels
    
    return augmented_data, augmented_labels

def main(feature_files, label_files, num_rotations, sampling_method):
    path = "/mnt/c/users/bonvi/Documents/deepset/data/"#'/work/mbonvin/data/'
    output_dir = os.path.join(path, 'augmented_data')
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if sampling_method == 'sobol':
        rotation_matrices = create_rot_mat_sobol(num_rotations)
    elif sampling_method == 'random':
        rotation_matrices = create_rot_mat_random(num_rotations)
    elif sampling_method == 'regular':
        rotation_matrices = create_rot_mat_regular(num_rotations)
    else:
        raise ValueError("Invalid sampling method. Choose 'sobol', 'random', or 'regular'.")

    for feature_file, label_file in zip(feature_files, label_files):
        input_feature_file = os.path.join(path, feature_file)
        input_label_file = os.path.join(path, label_file)

        # Open input files as memory-mapped arrays
        features = np.load(input_feature_file, mmap_mode='r')
        labels = np.load(input_label_file, mmap_mode='r')
        
        print(f"Processing {feature_file}")
        print(f"Loaded feature data shape: {features.shape}")
        print(f"Loaded label data shape: {labels.shape}")

        # Create memory-mapped output files
        output_feature_file = os.path.join(output_dir, f'augmented_features_{sampling_method}_{num_rotations}.npy')
        output_label_file = os.path.join(output_dir, f'augmented_labels_{sampling_method}_{num_rotations}.npy')
        
        augmented_features_shape = (features.shape[0] * num_rotations, features.shape[1], features.shape[2])
        augmented_labels_shape = (labels.shape[0] * num_rotations, labels.shape[1])
        
        augmented_features = np.lib.format.open_memmap(output_feature_file, mode='w+', dtype=features.dtype, shape=augmented_features_shape)
        augmented_labels = np.lib.format.open_memmap(output_label_file, mode='w+', dtype=labels.dtype, shape=augmented_labels_shape)

        # Process all data at once
        augmented_features[:], augmented_labels[:] = augment_data(features, labels, rotation_matrices)

        print(f"Augmentation complete. Augmented features saved to {output_feature_file}")
        print(f"Augmented labels saved to {output_label_file}")
        print(f"Final augmented feature shape: {augmented_features.shape}")
        print(f"Final augmented label shape: {augmented_labels.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory-mapped data augmentation with different SO(3) sampling methods')
    parser.add_argument('--sampling_method', type=str, choices=['sobol', 'random', 'regular'], default='sobol',
                        help='Sampling method for rotation matrices (default: sobol)')
    parser.add_argument('--num_rotations', type=int, default=8,
                        help='Number of rotations to generate (default: 8)')
    args = parser.parse_args()

    feature_files = [
        #'jetid_px_py_pz_eta_rel_phi_rel/processed/x_train_robust_8const_ptetaphi.npy',
        #'jetid_px_py_pz_eta_rel_phi_rel_16/processed/x_train_robust_16const_ptetaphi.npy',
        'jetid_px_py_pz_eta_rel_phi_rel_32/processed/x_train_robust_32const_ptetaphi.npy'
    ]
    label_files = [
        #'jetid_px_py_pz_eta_rel_phi_rel/processed/y_train_robust_8const_ptetaphi.npy',
        #'jetid_px_py_pz_eta_rel_phi_rel_16/processed/y_train_robust_16const_ptetaphi.npy',
        'jetid_px_py_pz_eta_rel_phi_rel_32/processed/y_train_robust_32const_ptetaphi.npy'
    ]
    
    main(feature_files, label_files, args.num_rotations, args.sampling_method)