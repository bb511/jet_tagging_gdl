import numpy as np
import torch
import os
import json
import pickle
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

class LogCountRobustScaler:
    def __init__(self, bins=200):
        self.bins = bins
        self.q1 = None
        self.q3 = None
        self.median = None

    def fit(self, data):
        data = data.flatten()
        hist, bin_edges = np.histogram(data, bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        log_counts = np.where(hist > 0, np.log(hist), 0.0)  # Assign log(0) for zero entries

        # Cumulative distribution over log-counts
        log_cdf = np.cumsum(log_counts)
        log_cdf /= log_cdf[-1]

        # Median
        median_idx = np.searchsorted(log_cdf, 0.5)
        self.median = bin_centers[median_idx]

        q25_idx = np.searchsorted(log_cdf, 0.25)
        q75_idx = np.searchsorted(log_cdf, 0.75)
        self.q3 = bin_centers[q75_idx]
        self.q1 = bin_centers[q25_idx]

    def transform(self, data):
        data = data.flatten()
        if self.q1 is None or self.q3 is None:
            return data
        iqr = self.q3 - self.q1
        return (data - self.median) / iqr
    
    def inverse_transform(self, data):
        data = data.flatten()
        if self.q1 is None or self.q3 is None:
            return data
        iqr = self.q3 - self.q1
        return (data * iqr + self.median).reshape(-1, 1)


    def __repr__(self):
        return f"LogCountRobustScaler(median={self.median:.3f}, q1={self.q1:.3f}, q3={self.q3:.3f})"
    
class MinMaxCustomScaler:
    def __init__(self, range=(-1,1)):
        
        self.range = range
        self.min = None
        self.max = None

    def fit(self, data):
        data = data.flatten()
        self.min = np.min(data)
        self.max = np.max(data)
       
    def transform(self, data):
        data = data.flatten()
        if self.min is None or self.max is None:
            return data
        if self.range == (-1,1):
            return 2 * data / (self.max - self.min) 
        elif self.range == (-2,2):
            return 4 * data / (self.max - self.min)
        elif self.range == (0,1):
            return data / (self.max - self.min)
        else:
            raise ValueError("Invalid range. Choose (-1, 1), (-2, 2), or (0, 1).")
    
    def inverse_transform(self, data):
        data = data.flatten()
        if self.min is None or self.max is None:
            return data
        if self.range == (-1, 1):
            return data* (self.max-self.min)/2
        elif self.range == (-2,2):
            return data* (self.max-self.min)/4
        elif self.range == (0,1):
            return data* (self.max-self.min)
        else:
            raise ValueError("Invalid range. Choose (-1, 1), (-2, 2), or (0, 1).")

    def __repr__(self):
        return f"MinMaxCustomScaler(min={self.min:.3f}, max={self.max:.3f})"
    

def fit_standardisation_old(method, data):
    """
    Fit the standardization parameters based on the chosen method.

    :param method: str, 'minmax', 'robust', or 'standard'
    :param data: numpy array of shape (n_samples, n_constituents, n_features)
    :return: dict with fitted scaler(s)
    """
    n_features = data.shape[2]
    params = {}

    for i in range(n_features):
        feature_data = data[:, :, i].reshape(-1, 1)
        
        if method == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(feature_data)
            params[f'scaler_{i}'] = scaler
        elif method == 'robust':
            scaler = RobustScaler()
            scaler.fit(feature_data)
            params[f'scaler_{i}'] = scaler
        elif method == 'standard':
            scaler = StandardScaler()
            scaler.fit(feature_data)
            params[f'scaler_{i}'] = scaler
        elif method == 'none':
            params[f'scaler_{i}'] = None
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax', 'robust', or 'standard'.")
        
    print("The normalisation parameters are: ")
    print("\n".join(f"{param}: {value}" for param, value in params.items()))
    print(f"{'':-<25}\n")

    return params

def fit_standardisation(method, data, feat="px_py_pz_etarel_phirel", consider_padding=True):
    """
    Fit the standardization parameters based on the chosen method, ignoring zero entries.

    :param method: str, 'minmax', 'robust', or 'standard'
    :param data: numpy array of shape (n_samples, n_constituents, n_features)
    :return: dict with fitted scaler(s)
    """
    n_features = data.shape[2]
    params = {}

    for i in range(n_features):
        feature_data = data[:, :, i].reshape(-1, 1)
        
        # Exclude zero entries
        if not consider_padding:
            feature_data = feature_data[feature_data != 0].reshape(-1, 1)
            
            if feature_data.size == 0:
                # If all entries are zero, skip fitting and set scaler to None
                print(f"Feature {i} contains only zeros. Skipping scaler fitting.")
                params[f'scaler_{i}'] = None
                continue
        
        if method == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(feature_data)
            params[f'scaler_{i}'] = scaler
        elif method == 'robust':
            scaler = RobustScaler()
            scaler.fit(feature_data)
            params[f'scaler_{i}'] = scaler
        elif method == 'standard':
            scaler = StandardScaler()
            scaler.fit(feature_data)
            params[f'scaler_{i}'] = scaler
        elif method == 'logcount_robust':
            scaler = LogCountRobustScaler()
            scaler.fit(feature_data)
            params[f'scaler_{i}'] = scaler
        elif method == 'minmaxcustom':
            continue      
        elif method == 'none':
            params[f'scaler_{i}'] = None
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax', 'robust', or 'standard'.")
    
    if method == 'minmaxcustom':
        # Fit the custom MinMax scaler with the specified range
        
        if feat == "px_py_pz_etarel_phirel":
            scaler = MinMaxCustomScaler(range=(-2, 2))
            scaler.fit(data[:, :, 2].reshape(-1, 1))
            params[f'scaler_{0}'] = scaler
            params[f'scaler_{1}'] = scaler
            params[f'scaler_{2}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1,1))
            scaler.fit(data[:, :, 3].reshape(-1, 1))
            params[f'scaler_{3}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1,1))
            scaler.fit(data[:, :, 4].reshape(-1, 1))
            params[f'scaler_{4}'] = scaler
            
        elif feat == "px_py_pz_pt_eta_etarel_phi_phirel":
            scaler = MinMaxCustomScaler(range=(-2, 2))
            scaler.fit(data[:, :, 2].reshape(-1, 1))
            params[f'scaler_{0}'] = scaler
            params[f'scaler_{1}'] = scaler
            params[f'scaler_{2}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(0, 1))
            scaler.fit(data[:, :, 3].reshape(-1, 1))
            params[f'scaler_{3}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1,1))
            scaler.fit(data[:, :, 4].reshape(-1, 1))
            params[f'scaler_{4}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1,1))
            scaler.fit(data[:, :, 5].reshape(-1, 1))
            params[f'scaler_{5}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1,1))
            scaler.fit(data[:, :, 6].reshape(-1, 1))
            params[f'scaler_{6}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1,1))
            scaler.fit(data[:, :, 7].reshape(-1, 1))
            params[f'scaler_{7}'] = scaler
            
        elif feat == "pt_etarel_phirel":
            scaler = MinMaxCustomScaler(range=(0, 1))
            scaler.fit(data[:, :, 0].reshape(-1, 1))
            params[f'scaler_{0}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1, 1))
            scaler.fit(data[:, :, 1].reshape(-1, 1))
            params[f'scaler_{1}'] = scaler
            
            scaler = MinMaxCustomScaler(range=(-1, 1))
            scaler.fit(data[:, :, 2].reshape(-1, 1))
            params[f'scaler_{2}'] = scaler
            
        elif feat in ["px_py_pz_jet_center", "px_py_pz"]:
            scaler = MinMaxCustomScaler(range=(-2, 2))
            scaler.fit(data[:, :, 2].reshape(-1, 1))
            params[f'scaler_{0}'] = scaler
            params[f'scaler_{1}'] = scaler
            params[f'scaler_{2}'] = scaler
        
        elif feat in ["px_py_jet_center","px_py"]:
            scaler = MinMaxCustomScaler(range=(-1, 1))
            scaler.fit(data[:, :, 0].reshape(-1, 1))
            params[f'scaler_{0}'] = scaler
            params[f'scaler_{1}'] = scaler     
              
    print("The normalization parameters are: ")
    print("\n".join(f"{param}: {value}" for param, value in params.items()))
    print(f"{'':-<25}\n")

    return params

def apply_standardisation_with_padding(data, params):
    """
    Apply the standardization to the data using pre-fitted parameters.

    :param data: numpy array or PyTorch tensor of shape (n_samples, n_constituents, n_features)
    :param params: dict with fitted scaler(s)
    :return: normalized data with the same shape as input
    """
    if params[f'scaler_0'] is None:
        normalized_data = data
        
    else:
        is_tensor = isinstance(data, torch.Tensor)
        
        # If the input is a PyTorch tensor, move it to CPU and convert to NumPy
        if is_tensor:
            device = data.device  # Save the device (CPU or GPU)
            data = data.cpu().numpy()  # Convert to NumPy array
        
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[2]):
            feature_data = data[:, :, i].reshape(-1, 1)
            scaler = params[f'scaler_{i}']
            normalized_feature = scaler.transform(feature_data)
            normalized_data[:, :, i] = normalized_feature.reshape(data.shape[0], data.shape[1])
        
        # If the input was a PyTorch tensor, convert back to tensor and move to the original device
        if is_tensor:
            normalized_data = torch.from_numpy(normalized_data).to(device)
    
    return normalized_data

def apply_standardisation(data, params, with_padding=True):
    """
    Apply the standardization to the data using pre-fitted parameters, ignoring zero values.

    :param data: numpy array or PyTorch tensor of shape (n_samples, n_constituents, n_features)
    :param params: dict with fitted scaler(s), one per feature as 'scaler_0', 'scaler_1', ...
    :return: normalized data with the same shape as input
    """
    is_tensor = isinstance(data, torch.Tensor)

    if is_tensor:
        device = data.device
        data = data.cpu().numpy()

    normalized_data = data.copy()

    for i in range(data.shape[2]):
        scaler_key = f'scaler_{i}'
        scaler = params.get(scaler_key, None)

        if scaler is None:
            continue  # Skip features with no scaler

        feature_data = data[:, :, i]
        
        if not with_padding:
            non_zero_mask = feature_data != 0
            if np.any(non_zero_mask):  # Only apply if non-zero entries exist
                non_zero_data = feature_data[non_zero_mask].reshape(-1, 1)
                normalized_non_zero = scaler.transform(non_zero_data).reshape(-1)
                normalized_data[:, :, i][non_zero_mask] = normalized_non_zero
            # Zeros stay untouched
        else:
            feature_data = feature_data.reshape(-1, 1)
            normalized_feature = scaler.transform(feature_data)
            normalized_data[:, :, i] = normalized_feature.reshape(data.shape[0], data.shape[1])

    if is_tensor:
        normalized_data = torch.from_numpy(normalized_data).to(device)

    return normalized_data

# def apply_standardisation(data, params):
#     """
#     Apply the standardization to the data using pre-fitted parameters, ignoring zero values.

#     :param data: numpy array or PyTorch tensor of shape (n_samples, n_constituents, n_features)
#     :param params: dict with fitted scaler(s)
#     :return: normalized data with the same shape as input
#     """
#     if params[f'scaler_0'] is None:
#         normalized_data = data
#     else:
#         is_tensor = isinstance(data, torch.Tensor)
        
#         # If the input is a PyTorch tensor, move it to CPU and convert to NumPy
#         if is_tensor:
#             device = data.device  # Save the device (CPU or GPU)
#             data = data.cpu().numpy()  # Convert to NumPy array
        
#         normalized_data = np.zeros_like(data)
#         for i in range(data.shape[2]):
#             feature_data = data[:, :, i]
            
#             # Create a mask for non-zero values
#             non_zero_mask = feature_data != 0
            
#             # Apply standardization only to non-zero values
#             scaler = params[f'scaler_{i}']
#             if scaler is not None:
#                 non_zero_data = feature_data[non_zero_mask].reshape(-1, 1)
#                 normalized_non_zero_data = scaler.transform(non_zero_data).reshape(-1)
#                 normalized_data[:, :, i][non_zero_mask] = normalized_non_zero_data
            
#             # Keep zero values as they are
#             normalized_data[:, :, i][~non_zero_mask] = 0
        
#         # If the input was a PyTorch tensor, convert back to tensor and move to the original device
#         if is_tensor:
#             normalized_data = torch.from_numpy(normalized_data).to(device)
    
#     return normalized_data

def inverse_standardisation(data, params, with_padding=True):
    """
    Apply the inverse standardization to the data using pre-fitted parameters.

    :param data: numpy array or PyTorch tensor of shape (n_samples, n_constituents, n_features)
    :param params: dict with fitted scaler(s)
    :return: denormalized data with the same shape as input
    """
    is_tensor = isinstance(data, torch.Tensor)

    if is_tensor:
        device = data.device
        data = data.cpu().numpy()

    denormalized_data = data.copy()

    for i in range(data.shape[2]):
        scaler_key = f'scaler_{i}'
        scaler = params.get(scaler_key, None)

        if scaler is None:
            continue  # skip if no scaler

        feature_data = data[:, :, i]
        non_zero_mask = feature_data != 0

        if not with_padding:
            if np.any(non_zero_mask):
                non_zero_data = feature_data[non_zero_mask].reshape(-1, 1)
                restored = scaler.inverse_transform(non_zero_data).reshape(-1)
                denormalized_data[:, :, i][non_zero_mask] = restored
                #zero stays zero by default
        else:
            feature_data = feature_data.reshape(-1, 1)
            restored = scaler.inverse_transform(feature_data)
            denormalized_data[:, :, i] = restored.reshape(data.shape[0], data.shape[1])

    if is_tensor:
        denormalized_data = torch.from_numpy(denormalized_data).to(device)

    return denormalized_data


# def inverse_standardisation(data, params):
#     """
#     Apply the inverse standardization to the data using pre-fitted parameters.

#     :param data: numpy array or PyTorch tensor of shape (n_samples, n_constituents, n_features)
#     :param params: dict with fitted scaler(s)
#     :return: denormalized data with the same shape as input
#     """
#     if params[f'scaler_0'] is None:
#         denormalized_data = data
        
#     else:
#         is_tensor = isinstance(data, torch.Tensor)
        
#         # If the input is a PyTorch tensor, move it to CPU and convert to NumPy
#         if is_tensor:
#             device = data.device  # Save the device (CPU or GPU)
#             data = data.cpu().numpy()  # Convert to NumPy array
        
#         denormalized_data = np.zeros_like(data)
        
#         for i in range(data.shape[2]):
#             feature_data = data[:, :, i].reshape(-1, 1)
#             scaler = params[f'scaler_{i}']
#             denormalized_feature = scaler.inverse_transform(feature_data)
#             denormalized_data[:, :, i] = denormalized_feature.reshape(data.shape[0], data.shape[1])
        
#         # If the input was a PyTorch tensor, convert back to tensor and move to the original device
#         if is_tensor:
#             denormalized_data = torch.from_numpy(denormalized_data).to(device)
    
#     return denormalized_data

def save_normalization_coeff(params, json_file):
    """
    Save normalization parameters to disk.
    :param data_name: str, name of the dataset
    :param params: dict with fitted scaler(s)
    """
    # Create directory if it doesn't exist
    # if not os.path.exists("normalization_coef"):
    #     os.makedirs("normalization_coef")
    
    # json_file = f"normalization_coef/normalization_parameters_{data_name}.json"
    # if os.path.exists(json_file):
    #     os.remove(json_file)
    
    # Extract the numerical parameters from each scaler
    json_params = {}
    for key, scaler in params.items():
        if scaler is None:
            json_params[key] = None
            continue
            
        scaler_params = {}
        if isinstance(scaler, MinMaxScaler):
            scaler_params['type'] = 'minmax'
            scaler_params['min'] = scaler.min_.tolist()
            scaler_params['scale'] = scaler.scale_.tolist()
            scaler_params['data_min'] = scaler.data_min_.tolist()
            scaler_params['data_max'] = scaler.data_max_.tolist()
        elif isinstance(scaler, StandardScaler):
            scaler_params['type'] = 'standard'
            scaler_params['mean'] = scaler.mean_.tolist()
            scaler_params['scale'] = scaler.scale_.tolist()
            scaler_params['var'] = scaler.var_.tolist()
        elif isinstance(scaler, RobustScaler):
            scaler_params['type'] = 'robust'
            scaler_params['center'] = scaler.center_.tolist()
            scaler_params['scale'] = scaler.scale_.tolist()
        
        json_params[key] = scaler_params
    
    with open(json_file, 'w') as f:
        print(f'Saving normalization parameters (JSON format) in {json_file}')
        json.dump(json_params, f)

def load_normalization_coeff(data_name, json_file):
    """
    Load normalization parameters from disk.
    :param data_name: str, name of the dataset
    :return: dict with fitted parameters
    """
    #file_name = f"normalization_coef/normalization_parameters_{data_name}.json"'
    file_name = json_file
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Normalization parameters for {data_name} not found.")
    
    with open(file_name, 'r') as f:
        print(f'Loading normalization parameters from {file_name}')
        params = json.load(f)
        
    print(params)
    
    return params