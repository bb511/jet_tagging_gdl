import numpy as np

def compute_mean_with_zero_handling(numerator, denominator, threshold=0):
    """Computes the mean of the ratio of two arrays, handling division by zero.
    Only valid (nonzero denominator) entries are included in the mean."""
    valid_mask = denominator > threshold
    ratio = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=ratio, where=valid_mask)
    valid_counts = np.sum(valid_mask, axis=1)
    ratio_sum = np.sum(ratio, axis=1)
    mean = np.divide(ratio_sum, valid_counts, out=np.zeros_like(ratio_sum, dtype=float), where=valid_counts > 0)
    return mean

def compute_mean_difference_with_zero_handling(minuend, subtrahend, constrained=False):
    """Computes the mean of the difference of two arrays, only over valid (nonzero subtrahend) entries."""
    valid_mask = (subtrahend != 0) & (minuend != 0)
    if constrained:
        diff = (minuend - subtrahend+np.pi)% (2 * np.pi) - np.pi
    else:
        diff = minuend - subtrahend
    # Set invalid entries to 0, but don't count them in the mean
    valid_counts = np.sum(valid_mask, axis=1)
    diff_sum = np.sum(np.where(valid_mask, diff, 0), axis=1)
    mean = np.divide(diff_sum, valid_counts, out=np.zeros_like(diff_sum, dtype=float), where=valid_counts > 0)
    return mean

def compute_jet_info(x_data: np.ndarray):
    """Computes the jet energy, transverse momentum, pseudo-rapidity, and azimuthal angle."""
    jet_E = compute_mean_with_zero_handling(x_data[:, :, 3], x_data[:, :, 4], threshold=1e-10)
    jet_pT = compute_mean_with_zero_handling(x_data[:, :, 5], x_data[:, :, 6])
    jet_eta = compute_mean_difference_with_zero_handling(x_data[:, :, 7], x_data[:, :, 8])
    jet_phi = (compute_mean_difference_with_zero_handling(x_data[:, :, 9], x_data[:, :, 10], constrained=True)) 
    
    return jet_E, jet_pT, jet_eta, jet_phi

def compute_center_xyz(x_data: np.ndarray, separated: bool = False):    
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    
    _, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    jet_px = jet_pT * np.cos(jet_phi)
    jet_py = jet_pT * np.sin(jet_phi)
    jet_pz = jet_pT * np.sinh(jet_eta) 
    
    jet_center = np.stack((jet_px, jet_py, jet_pz), axis=-1)
    if separated:
        # If separated, return jet center as separate features
        return jet_center
    else:
        jet_center = np.expand_dims(jet_center, axis=1)
        return np.concatenate((x_data[:, :, 0:3], jet_center), axis=1)
    
def compute_center_xyzE(x_data: np.ndarray, separated: bool = False):    
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    
    jet_E, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    jet_px = jet_pT * np.cos(jet_phi)
    jet_py = jet_pT * np.sin(jet_phi)
    jet_pz = jet_pT * np.sinh(jet_eta) 
    
    jet_center = np.stack((jet_px, jet_py, jet_pz, jet_E), axis=-1)
    if separated:
        # If separated, return jet center as separate features
        return jet_center
    else:
        jet_center = np.expand_dims(jet_center, axis=1)
        return np.concatenate((x_data[:, :, 0:4], jet_center), axis=1)
    
def compute_center_xy(x_data: np.ndarray, separated: bool = False):    
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    
    _, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    jet_px = jet_pT * np.cos(jet_phi)
    jet_py = jet_pT * np.sin(jet_phi)
    
    jet_center = np.stack((jet_px, jet_py), axis=-1)
    if separated:
        # If separated, return jet center as separate features
        return jet_center
    else:
        jet_center = np.expand_dims(jet_center, axis=1)
        return np.concatenate((x_data[:, :, 0:2], jet_center), axis=1)

def compute_center_xyz_bad(x_data: np.ndarray):
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    _, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    # Compute jet center in Cartesian coordinates
    jet_px = jet_pT * np.cos(jet_phi)
    jet_py = jet_pT * np.sin(jet_phi)
    jet_pz = jet_pT * np.sinh(jet_eta)
    
    # Stack jet center features
    jet_center = np.stack((jet_px, jet_py, jet_pz), axis=-1)  # Shape: (n_samples, 3)
    jet_center = np.expand_dims(jet_center, axis=1)  # Shape: (n_samples, 1, 3)
    jet_center = np.repeat(jet_center, x_data.shape[1], axis=1)  # Match n_constituents
    
    # Concatenate jet center as new features (extend last dimension)
    return np.concatenate((x_data, jet_center), axis=2)

def compute_center_pt_eta_phi(x_data: np.ndarray, separated: bool = False):
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    _, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    jet_center = np.stack((jet_pT, jet_eta, jet_phi), axis=-1)
    
    if separated:
        # If separated, return jet center as separate features
        return jet_center
    else:
        jet_center = np.expand_dims(jet_center, axis=1)
        return np.concatenate((x_data[:, :, [3, 5, 7]], jet_center), axis=1)
    
def compute_center_xyz_pt_eta_phi(x_data: np.ndarray, separated: bool = False):
    jet_center_pt_eta_phi = compute_center_pt_eta_phi(x_data, separated=True)
    jet_center_xyz = compute_center_xyz(x_data, separated=True)
    jet_center = np.concatenate((jet_center_pt_eta_phi, jet_center_xyz), axis=-1)
    if separated:
        # If separated, return jet center as separate features
        return jet_center
    else:
        jet_center = np.expand_dims(jet_center, axis=1)
        return np.concatenate((x_data[:, :, [0, 1, 2, 3, 5, 7]], jet_center), axis=1)
    


