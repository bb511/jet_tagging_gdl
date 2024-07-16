import numpy as np
import os
from tqdm import tqdm

def create_rot_mat(theta, phi, psi):
    """Function to create rotation matrices from the Euler angles using NumPy"""
    dic_rot = {}
    
    def create_single_rot_mat(t, p, s):
        Rz = np.array([[np.cos(t), -np.sin(t), 0],
                       [np.sin(t), np.cos(t), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)],
                       [0, 1, 0],
                       [-np.sin(p), 0, np.cos(p)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(s), -np.sin(s)],
                       [0, np.sin(s), np.cos(s)]])
        return Rz @ Ry @ Rx

    # Main rotation matrix
    dic_rot['R'] = create_single_rot_mat(theta, phi, psi)

    # Additional rotation matrices
    angles = np.linspace(0, 2*np.pi, 8)
    for angle_theta in angles:
        for angle_phi in angles:
            for angle_psi in angles:
                key = f'{angle_theta:.4f}, {angle_phi:.4f}, {angle_psi:.4f}'
                dic_rot[key] = create_single_rot_mat(theta + angle_theta, 
                                                     phi + angle_phi, 
                                                     psi + angle_psi)

    # Identity matrix
    dic_rot['Id'] = np.eye(3)

    return dic_rot

def augment_data(data, rotation_matrices):
    """Apply rotations to the data"""
    augmented_data = []
    for rotation_key, rotation_matrix in rotation_matrices.items():
        rotated_data = np.dot(data, rotation_matrix.T)
        augmented_data.append(rotated_data)
    return np.array(augmented_data)

def main():
    # Load the original data
    input_file = 'original_data.npy'
    output_dir = 'augmented_data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load(input_file)
    print(f"Loaded data shape: {data.shape}")

    # Generate rotation matrices
    theta, phi, psi = np.pi/4, np.pi/3, np.pi/6 
    rotation_matrices = create_rot_mat(theta, phi, psi)

    # Augment data batch by batch
    for i in tqdm(range(data.shape[0]), desc="Augmenting data"):
        batch_data = data[i]
        augmented_batch = augment_data(batch_data, rotation_matrices)
        
        # Save each augmented batch
        for j, aug_data in enumerate(augmented_batch):
            output_file = os.path.join(output_dir, f'augmented_batch_{i}_rotation_{j}.npy')
            np.save(output_file, aug_data)

    print(f"Augmentation complete. Augmented data saved in {output_dir}")

if __name__ == "__main__":
    main()