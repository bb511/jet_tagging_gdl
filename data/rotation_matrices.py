import numpy as np
from scipy.stats import qmc
from scipy.spatial.transform import Rotation as R
import torch

def create_rot_mat_sobol(num_rotations):
    sobol_engine = qmc.Sobol(d=4, scramble=True)
    sobol_samples = sobol_engine.random(n=num_rotations)
    
    rot_matrices = []
    for sample in sobol_samples:
        q = sample_to_quaternion(sample)
        rot_matrices.append(quaternion_to_rotation_matrix(q))
    
    return np.array(rot_matrices)

def create_rot_matrices_scipy(nbr_rot: int, seed: int = None):
    """Generate random rotation matrices in SO(3) using scipy with an optional seed."""
    rot = R.random(nbr_rot, random_state=seed)  # Generate `nbr_rot` random rotations with a seed
    return rot.as_matrix()  # Convert to rotation matrices

def create_rot_mat_random(num_rotations, seed=None):
    rng = np.random.default_rng(seed)
    rot_matrices = []
    for _ in range(num_rotations):
        A = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(A)
        rot_matrices.append(Q * np.sign(np.diag(R)))
    return np.array(rot_matrices)

def create_rot_mat_regular(num_rotations):
    rot_matrices = []
    angles = np.linspace(0, 2*np.pi, int(np.cbrt(num_rotations)), endpoint=False)
   
    for theta in angles:
        for phi in angles:
            for psi in angles:
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
   
    return np.array(rot_matrices[:num_rotations])

def sample_to_quaternion(sample):
    theta1 = 2 * np.pi * sample[0]
    theta2 = 2 * np.pi * sample[1]
    u = sample[2]
    v = sample[3]
    
    q = np.array([
        np.sqrt(1 - u) * np.sin(theta1),
        np.sqrt(1 - u) * np.cos(theta1),
        np.sqrt(u) * np.sin(theta2),
        np.sqrt(u) * np.cos(theta2)
    ])
    
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
def create_so2_rot_matrices(nbr_rot: int, seed: int = None):
    """Generate random SO(2) rotation matrices in the xy-plane (3x3 matrices)."""
    seed = seed if seed is not None else np.random.randint(0, 1e6)
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, nbr_rot)  # random angles [0, 2pi)

    rot_mats = np.zeros((nbr_rot, 3, 3), dtype=np.float32)

    rot_mats[:, 0, 0] = np.cos(thetas)
    rot_mats[:, 0, 1] = -np.sin(thetas) 
    rot_mats[:, 1, 0] = np.sin(thetas)
    rot_mats[:, 1, 1] = np.cos(thetas)
    rot_mats[:, 2, 2] = 1.0  # pz unchanged

    return rot_mats

def create_Lorentz_mat(nbr_rot:int, device):
    """Function to create Lorentz matrices using characterization of Lorentz transformations"""
    rot_mats = []
    for _ in range(16):
        R = np.random.rand(3, 3)
        U, _, V = np.linalg.svd(R)
        #generate two elements of SO(3)
        P = U@U
        Q = V@V
        M = np.zeros((4,4))
        M[:-1, :-1] = P
        M[-1,-1]= 1
        M = torch.tensor(M).to(torch.float32).to(device)
        N = np.zeros((4,4))
        N[:-1, :-1] = np.transpose(Q)
        N[-1,-1]= 1
        N = torch.tensor(N).to(torch.float32).to(device)
        rot_mats.append((M,N))
    return rot_mats