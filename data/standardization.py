import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def fit_standardisation(method, data):
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
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax', 'robust', or 'standard'.")
        
        scaler.fit(feature_data)
        params[f'scaler_{i}'] = scaler
        
    print("The normalisation parameters are: ")
    print("\n".join(f"{param}: {value}" for param, value in params.items()))
    print(f"{'':-<25}\n")

    return params

def apply_standardisation(data, params):
    """
    Apply the standardization to the data using pre-fitted parameters.

    :param data: numpy array of shape (n_samples, n_constituents, n_features)
    :param params: dict with fitted scaler(s)
    :return: normalized data with the same shape as input
    """
    normalized_data = np.zeros_like(data)
    
    for i in range(data.shape[2]):
        feature_data = data[:, :, i].reshape(-1, 1)
        scaler = params[f'scaler_{i}']
        normalized_feature = scaler.transform(feature_data)
        normalized_data[:, :, i] = normalized_feature.reshape(data.shape[0], data.shape[1])
    
    return normalized_data

def inverse_standardisation(data, params):
    """
    Apply the inverse standardization to the data using pre-fitted parameters.

    :param data: numpy array of shape (n_samples, n_constituents, n_features)
    :param params: dict with fitted scaler(s)
    :return: denormalized data with the same shape as input
    """
    denormalized_data = np.zeros_like(data)
    
    for i in range(data.shape[2]):
        feature_data = data[:, :, i].reshape(-1, 1)
        scaler = params[f'scaler_{i}']
        denormalized_feature = scaler.inverse_transform(feature_data)
        denormalized_data[:, :, i] = denormalized_feature.reshape(data.shape[0], data.shape[1])
    
    return denormalized_data