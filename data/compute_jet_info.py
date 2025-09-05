import numpy as np

# def compute_mean_with_zero_handling(numerator, denominator, threshold=0):
#     """Computes the mean of the ratio of two arrays, handling division by zero."""
#     valid_mask = denominator > threshold
#     result = np.zeros_like(numerator)
#     np.divide(numerator, denominator, out=sresult, where=valid_mask)
#     return np.mean(result, axis=1)

# def compute_mean_difference_with_zero_handling(minuend, subtrahend):
#     """Computes the mean of the difference of two arrays, handling subtraction by zero."""
#     valid_mask = subtrahend != 0
#     result = np.where(valid_mask, minuend - subtrahend, 0)
#     return np.mean(result, axis=1)

# def compute_jet_info(x_data: np.ndarray):
#     """Computes the jet energy, transverse momentum, pseudo-rapidity, and azimuthal angle."""
#     jet_E = compute_mean_with_zero_handling(x_data[:, :, 3], x_data[:, :, 4], threshold=1e-10)
#     jet_pT = compute_mean_with_zero_handling(x_data[:, :, 5], x_data[:, :, 6])
#     jet_eta = compute_mean_difference_with_zero_handling(x_data[:, :, 7], x_data[:, :, 8])
#     jet_phi = compute_mean_difference_with_zero_handling(x_data[:, :, 9], x_data[:, :, 10])
    
#     return jet_E, jet_pT, jet_eta, jet_phi
    
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
    
def compute_center_energy(x_data: np.ndarray):
    """Computes the jet center based on the transverse momentum, pseudo-rapidity, and azimuthal angle."""
    jet_E, jet_pT, jet_eta, jet_phi = compute_jet_info(x_data)
    
    jet_px = jet_pT * np.cos(jet_phi)
    jet_py = jet_pT * np.sin(jet_phi)
    jet_pz = jet_pT * np.sinh(jet_eta)
    
    jet_center = np.stack((jet_px, jet_py, jet_pz, jet_E), axis=-1)
    jet_center = np.expand_dims(jet_center, axis=1)
    
    return np.concatenate((x_data[:, :, 0:4], jet_center), axis=1)

def compute_ptrelinv(x_data: np.ndarray):
    """
    Computes the relative transverse momentum of each particle w.r.t. the jet axis.
    x_data is expected to have shape (n_samples, n_constituents, n_features),
    where features [0, 1, 2] are px, py, pz.
    """
    # Extract particle momentum
    part_p = x_data[:, :, 0:3]  # (N, n_const, 3)

    # Compute jet axis in Cartesian coordinates (px, py, pz)
    jet_center = compute_center_xyz(x_data, separated=True)  # (N, 3)
    jet_center_norm = np.linalg.norm(jet_center, axis=1, keepdims=True) + 1e-10  # (N, 1) to prevent div by 0
    jet_axis = jet_center / jet_center_norm  # (N, 3)

    # Expand jet_axis to match particle dimensions
    jet_axis = np.expand_dims(jet_axis, axis=1)  # (N, 1, 3)
    
    # Compute cross product of each particle momentum with jet axis (broadcasted)
    cross_prod = np.cross(part_p, jet_axis)  # (N, n_const, 3)
    
    # Norm of cross product gives pT^rel
    pT_rel = np.linalg.norm(cross_prod, axis=2)  # (N, n_const)
    
    return pT_rel

def compute_etarelinv(x_data: np.ndarray):
    """
    Computes the relative pseudo-rapidity of each particle w.r.t. the jet axis.
    """
    part_p = x_data[:, :, 0:3]  # (N, n_const, 3)
    jet_center = compute_center_xyz(x_data, separated=True)  # (N, 3)
    jet_axis = jet_center / (np.linalg.norm(jet_center, axis=1, keepdims=True) + 1e-10)
    jet_axis = np.expand_dims(jet_axis, axis=1)  # (N, 1, 3)

    p_par = np.sum(part_p * jet_axis, axis=2)       # parallel component
    p_mag = np.linalg.norm(part_p, axis=2)          # magnitude of particle momenta

    eta_rel = 0.5 * np.log((p_mag + p_par + 1e-10) / (p_mag - p_par + 1e-10))

    return eta_rel  # shape (N, n_const)

def build_jet_frame_geometric(x_data: np.ndarray):
    """
    Builds a right-handed jet-aligned frame using the particle with highest pT^rel
    as transverse reference. Returns (e1, e2), each of shape (N, 3).
    
    x_data is (N, n_const, n_feat), where features [0,1,2] are px,py,pz
    """
    # Compute jet axis
    jet_center = compute_center_xyz(x_data, separated=True)  # (N, 3)
    jet_axis = jet_center / (np.linalg.norm(jet_center, axis=1, keepdims=True) + 1e-10)  # (N, 3)

    # Compute pT^rel using given function
    pT_rel = compute_ptrelinv(x_data)  # (N, n_const)
    leading_idx = np.argmax(pT_rel, axis=1)  # (N,)

    # Extract 3-momentum of all particles
    part_p = x_data[:, :, 0:3]  # (N, n_const, 3)

    # Project leading particle onto transverse plane
    dot_n = np.sum(part_p * jet_axis[:, None, :], axis=2, keepdims=True)
    p_perp = part_p - dot_n * jet_axis[:, None, :]  # (N, n_const, 3)
    e1 = np.array([p_perp[n, leading_idx[n]] for n in range(x_data.shape[0])])
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-10)  # (N, 3)

    # Build right-handed orthonormal basis
    e2 = np.cross(jet_axis, e1)  # (N, 3)
    return e1, e2

def compute_phirelinv(x_data: np.ndarray):
    """
    Computes phi_rel of each particle w.r.t. the jet-aligned frame defined by the leading pT^rel particle.
    Returns shape (N, n_const) with phi values in [-pi, pi].
    """
    # Extract momenta
    part_p = x_data[:, :, 0:3]  # (N, n_const, 3)
    jet_center = compute_center_xyz(x_data, separated=True)  # (N, 3)
    jet_axis = jet_center / (np.linalg.norm(jet_center, axis=1, keepdims=True) + 1e-10)  # (N, 3)

    # Get orthonormal basis in transverse plane
    e1, e2 = build_jet_frame_geometric(x_data)  # (N, 3) each
    e1 = e1[:, None, :]  # (N, 1, 3)
    e2 = e2[:, None, :]  # (N, 1, 3)

    # Project part_p onto transverse plane
    dot_n = np.sum(part_p * jet_axis[:, None, :], axis=2, keepdims=True)
    p_perp = part_p - dot_n * jet_axis[:, None, :]  # (N, n_const, 3)

    # Coordinates in jet frame
    x = np.sum(p_perp * e1, axis=2)
    y = np.sum(p_perp * e2, axis=2)

    # Compute phi_rel
    phi_rel = np.arctan2(y, x)  # (N, n_const)
    return phi_rel

def compute_ptrelinv_etarelinv_phirelinv(x_data: np.ndarray):
    """
    Computes pT^rel, eta^rel, and phi^rel of each particle w.r.t. the jet axis.
    Returns shape (N, n_const, 3) with features [pT^rel, eta^rel, phi^rel].
    """
    pT_rel = compute_ptrelinv(x_data)  # (N, n_const)
    eta_rel = compute_etarelinv(x_data)  # (N, n_const)
    phi_rel = compute_phirelinv(x_data)  # (N, n_const)

    return np.stack((pT_rel, eta_rel, phi_rel), axis=-1)  # (N, n_const, 3)

def compute_xyz_etarelinv_phirelinv(x_data:np.ndarray):
    """
    Computes pT^rel, eta^rel, and phi^rel of each particle w.r.t. the jet axis.
    Returns shape (N, n_const, 3) with features [px, py, pz].
    """
    part_p = x_data[:, :, 0:3]  # (N, n_const, 3)
    eta_rel = compute_etarelinv(x_data)  # (N, n_const)
    phi_rel = compute_phirelinv(x_data)  # (N, n_const)

    return np.concatenate([part_p, eta_rel[..., None], phi_rel[..., None]], axis=-1)
    