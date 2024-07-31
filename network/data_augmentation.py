import numpy as np
import os
from tqdm import tqdm

def create_rot_mat():
    """Function to create 8 regularly rotated matrices"""
    rot_matrices = []
    angles = np.linspace(0, 2*np.pi, 2, endpoint=False)
   
    for theta in angles:  # rotation around z-axis
        for phi in angles:  # rotation around y-axis
            for psi in angles:  # rotation around x-axis
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
   
    return rot_matrices

def augment_data(data, labels, rotation_matrices):
    """Apply rotations to the data and duplicate labels"""
    augmented_data = []
    augmented_labels = []
    data_px_py_pz = data[:, :3]
    data_pt_eta_rel_phi_rel = data[:, 3:]
    for rotation_matrix in rotation_matrices:
        rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
        rotated_data = np.concatenate((rotated_data, data_pt_eta_rel_phi_rel), axis=1)
        augmented_data.append(rotated_data)
        augmented_labels.append(labels)
    return np.array(augmented_data), np.array(augmented_labels)

def main(feature_files, label_files):
    path = '/work/mbonvin/data/'
    output_dir = os.path.join(path, 'augmented_data')
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_augmented_features = []
    all_augmented_labels = []

    rotation_matrices = create_rot_mat()

    for feature_file, label_file in zip(feature_files, label_files):
        input_feature_file = os.path.join(path, feature_file)
        input_label_file = os.path.join(path, label_file)

        features = np.load(input_feature_file)
        labels = np.load(input_label_file)
        print(f"Processing {feature_file}")
        print(f"Loaded feature data shape: {features.shape}")
        print(f"Loaded label data shape: {labels.shape}")

        for i in tqdm(range(features.shape[0]), desc="Augmenting data"):
            batch_features = features[i]
            batch_labels = labels[i]
            augmented_batch_features, augmented_batch_labels = augment_data(batch_features, batch_labels, rotation_matrices)
            all_augmented_features.append(augmented_batch_features)
            all_augmented_labels.append(augmented_batch_labels)

    # Concatenate all augmented data
    all_augmented_features = np.concatenate(all_augmented_features, axis=0)
    all_augmented_labels = np.concatenate(all_augmented_labels, axis=0)

    # Save all augmented data to single files
    output_feature_file = os.path.join(output_dir, 'all_augmented_features_8.npy')
    output_label_file = os.path.join(output_dir, 'all_augmented_labels_8.npy')
    np.save(output_feature_file, all_augmented_features)
    np.save(output_label_file, all_augmented_labels)

    print(f"Augmentation complete. All augmented features saved to {output_feature_file}")
    print(f"All augmented labels saved to {output_label_file}")
    print(f"Final augmented feature shape: {all_augmented_features.shape}")
    print(f"Final augmented label shape: {all_augmented_labels.shape}")

if __name__ == "__main__":
    feature_files = [
        'jetid_px_py_pz_eta_rel_phi_rel/processed/x_train_robust_8const_ptetaphi.npy',
        #'jetid_px_py_pz_eta_rel_phi_rel_16/processed/x_train_robust_16const_ptetaphi.npy',
        #'jetid_px_py_pz_eta_rel_phi_rel_32/processed/x_train_robust_32const_ptetaphi.npy'
    ]
    label_files = [
        'jetid_px_py_pz_eta_rel_phi_rel/processed/y_train_robust_8const_ptetaphi.npy',
        #'jetid_px_py_pz_eta_rel_phi_rel_16/processed/y_train_robust_16const_ptetaphi.npy',
        #'jetid_px_py_pz_eta_rel_phi_rel_32/processed/y_train_robust_32const_ptetaphi.npy'
    ]
    
    main(feature_files, label_files)