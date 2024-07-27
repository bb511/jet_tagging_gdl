import numpy as np
import glob

def concatenate_npy_files(file_pattern, output_file):
    # Get a list of all files matching the pattern
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No files found matching the pattern: {file_pattern}")
        return
    
    arrays = [np.load(file) for file in files]
    concatenated_array = np.concatenate(arrays, axis=0)
    
    np.save(output_file, concatenated_array)
    print(f"Concatenated array saved to: {output_file}")
    print(f"Shape of concatenated array: {concatenated_array.shape}")

def concatenate_npy_files_memory_efficient(file_pattern, output_file, chunk_size=1000):
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No files found matching the pattern: {file_pattern}")
        return
    
    first_array = np.load(files[0], mmap_mode='r')
    dtype = first_array.dtype
    shape = list(first_array.shape)
    
    total_items = sum(np.load(f, mmap_mode='r').shape[0] for f in files)
    shape[0] = total_items
    
    with open(output_file, 'wb') as f:
        np.lib.format.write_array_header_1_0(f, shape, dtype)
    
    with open(output_file, 'ab') as f:
        for file in files:
            array = np.load(file, mmap_mode='r')
            for i in range(0, array.shape[0], chunk_size):
                chunk = array[i:i+chunk_size]
                chunk.tofile(f)
    
    print(f"Concatenated array saved to: {output_file}")
    print(f"Shape of concatenated array: {tuple(shape)}")


# Example usage
file_pattern = "data_*.npy"  
output_file = "concatenated_data.npy"

concatenate_npy_files(file_pattern, output_file)