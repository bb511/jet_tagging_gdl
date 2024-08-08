import numpy as np
from sklearn.preprocessing import RobustScaler
import os

def calculate_statistics(file_path, shape):
    """
    Calculate statistics needed for RobustScaler from the training file.
    
    :param file_path: Path to the training .npy file
    :param shape: Shape of the file (n_point_clouds, n_vertices, n_features)
    :return: Fitted RobustScaler
    """
    n_features = shape[2]
    
    # Load the entire training data
    data = np.load(file_path, mmap_mode='r')
    data_reshaped = data.reshape(-1, n_features)
    
    # Fit the RobustScaler
    scaler = RobustScaler()
    scaler.fit(data_reshaped)
    
    return scaler

def normalize_data(input_path, output_path, scaler):
    """
    Normalize data using a pre-fitted RobustScaler and memory mapping.
    
    :param input_path: Path to input .npy file
    :param output_path: Path to output normalized .npy file
    :param scaler: Pre-fitted RobustScaler
    """
    input_data = np.load(input_path, mmap_mode='r')
    shape = input_data.shape
    
    output_data = np.memmap(output_path, dtype='float32', mode='w+', shape=shape)
    
    normalized_data = scaler.transform(input_data.reshape(-1, shape[2]))
    output_data[:] = normalized_data.reshape(shape)
    
    # Flush changes to disk
    output_data.flush()
    del output_data

def main():
    PATH = "/mnt/c/Users/bonvi/Desktop/jetclass_data/"
    data_types = ['x', 'jet']
    splits = ['train', 'test', 'valid']
    
    for data_type in data_types:
        print(f"Processing {data_type} files...")
        
        train_path = os.path.join(PATH, f"{data_type}_train_norm.npy")
        if not os.path.exists(train_path):
            print(f"Training file not found: {train_path}")
            continue
        
        print(f"Using training file: {train_path}")
        
        shape = np.load(train_path, mmap_mode='r').shape
        print(f"Shape of training file: {shape}")
        
        scaler = calculate_statistics(train_path, shape)
        
        for split in splits:
            input_path = os.path.join(PATH, f"{data_type}_{split}_norm.npy")
            if not os.path.exists(input_path):
                print(f"File not found: {input_path}")
                continue
            
            output_path = input_path.replace('_norm.npy', '_normalized.npy')
            print(f"Normalizing {input_path}...")
            normalize_data(input_path, output_path, scaler)
            print(f"Normalized data saved to {output_path}")
    
    print("Normalization complete.")

if __name__ == "__main__":
    main()