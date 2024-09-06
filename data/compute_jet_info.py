import numpy as np

def compute_mean_with_zero_handling(numerator, denominator, threshold=0):
    """Computes the mean of the ratio of two arrays, handling division by zero."""
    valid_mask = denominator > threshold
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=result, where=valid_mask)
    return np.mean(result, axis=1)

def compute_mean_difference_with_zero_handling(minuend, subtrahend):
    """Computes the mean of the difference of two arrays, handling subtraction by zero."""
    valid_mask = subtrahend != 0
    result = np.where(valid_mask, minuend - subtrahend, 0)
    return np.mean(result, axis=1)

def compute_jet_info(x_data: np.ndarray):
    """Computes the jet energy, transverse momentum, pseudo-rapidity, and azimuthal angle."""
    jet_E = compute_mean_with_zero_handling(x_data[:, :, 3], x_data[:, :, 4], threshold=1e-10)
    jet_pT = compute_mean_with_zero_handling(x_data[:, :, 5], x_data[:, :, 6])
    jet_eta = compute_mean_difference_with_zero_handling(x_data[:, :, 7], x_data[:, :, 8])
    jet_phi = compute_mean_difference_with_zero_handling(x_data[:, :, 9], x_data[:, :, 10])
    
    return jet_E, jet_pT, jet_eta, jet_phi

def compute_center_xyz(x_data: np.ndarray):
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    _, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    jet_px = jet_pT * np.cos(jet_phi)
    jet_py = jet_pT * np.sin(jet_phi)
    jet_pz = jet_pT * np.sinh(jet_eta)
    
    jet_center = np.stack((jet_px, jet_py, jet_pz), axis=-1)
    jet_center = np.expand_dims(jet_center, axis=1)
    
    return np.concatenate((x_data[:, :, 0:3], jet_center), axis=1)

def compute_center_pt_eta_phi(x_data: np.ndarray):
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    _, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    jet_center = np.stack((jet_pT, jet_eta, jet_phi), axis=-1)
    jet_center = np.expand_dims(jet_center, axis=1)
    
    return np.concatenate((x_data[:, :, [3, 5, 7]], jet_center), axis=1)

def compute_center_energy(x_data: np.ndarray):
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    jet_E, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    jet_px = jet_pT * np.cos(jet_phi)
    jet_py = jet_pT * np.sin(jet_phi)
    jet_pz = jet_pT * np.sinh(jet_eta)
    
    jet_center = np.stack((jet_px, jet_py, jet_pz, jet_E), axis=-1)
    jet_center = np.expand_dims(jet_center, axis=1)
    
    return np.concatenate((x_data[:, :, 0:4], jet_center), axis=1)