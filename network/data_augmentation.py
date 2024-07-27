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

def augment_data(data, rotation_matrices):
    """Apply rotations to the data"""
    augmented_data = []
    for rotation_matrix in rotation_matrices:
        rotated_data = np.dot(data, rotation_matrix.T)
        augmented_data.append(rotated_data)
    return np.array(augmented_data)

def main():
    path = 'C://Users//bonvi//Documents//deepset//data//'
    input_file = path + 'original_data.npy'
    output_dir = 'augmented_data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load(input_file)
    print(f"Loaded data shape: {data.shape}")

    rotation_matrices = create_rot_mat()

    for i in tqdm(range(data.shape[0]), desc="Augmenting data"):
        batch_data = data[i]
        augmented_batch = augment_data(batch_data, rotation_matrices)
        
        for j, aug_data in enumerate(augmented_batch):
            output_file = os.path.join(output_dir, f'augmented_batch_{i}_rotation_{j}.npy')
            np.save(output_file, aug_data)

    print(f"Augmentation complete. Augmented data saved in {output_dir}")

if __name__ == "__main__":
    main()