# Description: Contains custom loss functions for invariant neural networks.
# In particular, it contains functions to calculate the invariant error for a given model and input data.
# SO(3) loss for px_py_pz and px_py_pz_eta_rel_phi_rel data
# For regularly spaced SO(3) samples, we use the Euler angles to generate the rotation matrices.
# For Latin Hypercube samples, we use the axis-angle representation to generate the rotation matrices.
# For Sobol samples, we use the quaternion representation to generate the rotation matrices.
# For random samples, we generate random rotation matrices.
# For Lorentz transformations, we generate random rapidity values and use them to generate the Lorentz transformation matrices.
# For permutation, we permute the constituents of the input data.
# For phi rotations, we rotate the input data in phi.


import torch
from torch.nn import LogSoftmax
import numpy as np
import math
from scipy.stats import qmc

import util


# apparently one can use torch-geometric.transforms.RandomRotate to rotate the input data in a random way (and it should be efficient)
def inv_err_deterministic_eta_phi(model, criterion, x, y_pred, y_true, l, nbr_sym_loss, device):
    input = x
    assert x.shape[2] == 5, "The number of features should be 5"
    
    rot_mats = generate_regular_so3_samples(nbr_sym_loss, device)
    
    err = torch.zeros(rot_mats.shape[0])
    for i in range(rot_mats.shape[0]):
        a = y_pred
        px_py_pz = input[:, :, :3]
        rot_input = px_py_pz @ rot_mats[i]
        concat_input = torch.cat((rot_input, input[:,:,3:]), dim=2)
        b = model(concat_input)
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim=1))

def generate_regular_so3_samples(num_samples, device):
    samples_per_angle = int(np.cbrt(num_samples))
    
    alpha = np.linspace(0, 2*np.pi, samples_per_angle)
    beta = np.linspace(0, np.pi, samples_per_angle)
    gamma = np.linspace(0, 2*np.pi, samples_per_angle)
    
    rot_mats = []
    for a in alpha:
        for b in beta:
            for g in gamma:
                R = euler_angles_to_rotation_matrix(a, b, g)
                rot_mats.append(R)
    
    rot_mats = torch.stack([torch.tensor(R, dtype=torch.float32) for R in rot_mats]).to(device)
    
    if rot_mats.shape[0] > num_samples:
        indices = torch.randperm(rot_mats.shape[0])[:num_samples]
        rot_mats = rot_mats[indices]
    
    return rot_mats

def euler_angles_to_rotation_matrix(alpha, beta, gamma):

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    

    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    

    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    return R

def inv_err_lhs_eta_phi(model, criterion, x, y_pred, y_true, l, nbr_sym_loss, device):
    input = x
    assert x.shape[2] == 5, "The number of features should be 5"
    
    rot_mats = generate_lhs_so3_samples(nbr_sym_loss, device)
    
    err = torch.zeros(rot_mats.shape[0])
    for i in range(rot_mats.shape[0]):
        a = y_pred
        px_py_pz = input[:, :, :3]
        rot_input = px_py_pz @ rot_mats[i]
        concat_input = torch.cat((rot_input, input[:,:,3:]), dim=2)
        b = model(concat_input)
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim=1))

def generate_lhs_so3_samples(num_samples, device):
    sampler = qmc.LatinHypercube(d=4)
    
    lhs_samples = sampler.random(n=num_samples)
    
    rot_mats = []
    for sample in lhs_samples:
        alpha, beta, gamma = sample[:3] * 2 * np.pi
        
        angle = sample[3] * 2 * np.pi
        
        # Generate rotation axis
        axis = np.array([np.sin(alpha) * np.cos(beta),
                         np.sin(alpha) * np.sin(beta),
                         np.cos(alpha)])
        
        R = axis_angle_to_rotation_matrix(axis, angle)
        rot_mats.append(R)
    
    rot_mats = torch.stack([torch.tensor(R, dtype=torch.float32) for R in rot_mats]).to(device)
    
    return rot_mats

def axis_angle_to_rotation_matrix(axis, angle):
    """Convert axis-angle representation to rotation matrix using Rodrigues' rotation formula."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    
def inv_err_sobol_eta_phi(model, criterion, x, y_pred, y_true, l, nbr_sym_loss, device):
    input = x
    assert x.shape[2] == 5, "The number of features should be 5"
    
    # Generate rotation matrices using PyTorch Sobol sequence
    rot_mats = generate_sobol_so3_samples(nbr_sym_loss, device)
    
    err = torch.zeros(nbr_sym_loss, device=device)
    for i in range(nbr_sym_loss):
        a = y_pred
        px_py_pz = input[:, :, :3]
        rot_input = torch.bmm(px_py_pz, rot_mats[i].unsqueeze(0).expand(px_py_pz.size(0), -1, -1))
        concat_input = torch.cat((rot_input, input[:,:,3:]), dim=2)
        b = model(concat_input)
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim=1))

def generate_sobol_so3_samples(num_samples, device):
    sobol_engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True)
    
    sobol_samples = sobol_engine.draw(num_samples).to(device)
    
    rot_mats = sample_to_rot_matrix(sobol_samples)
    
    return rot_mats

def sample_to_rot_matrix(samples):
    x0 = samples[:, 0]
    x1 = samples[:, 1]
    x2 = samples[:, 2]
    x3 = samples[:, 3]

    r1 = torch.sqrt(1.0 - x0)
    r2 = torch.sqrt(x0)
    t1 = 2 * math.pi * x1
    t2 = 2 * math.pi * x2
    cos1 = torch.cos(t1)
    sin1 = torch.sin(t1)
    cos2 = torch.cos(t2)
    sin2 = torch.sin(t2)

    q = torch.stack([r1 * sin1, r1 * cos1, r2 * sin2, r2 * cos2], dim=1)
    
    return quaternion_to_rotation_matrix(q)

def quaternion_to_rotation_matrix(q):
    q = q / torch.norm(q, dim=1, keepdim=True)
    
    w, x, y, z = q.t()
    
    rot_matrix = torch.stack([
        1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y
    ], dim=1).reshape(-1, 3, 3)
    
    return rot_matrix


def inv_err_rand(model, criterion, x, y_pred, y_true, l, nbr_rot, device):
    """ Custom loss function that checks the invariance of the model to rotations
        It does it using randomly generated rotation matrices"""
    model.eval()
    logsoftmax = LogSoftmax(dim=1) #numerically more stable than softmax
    
    input = x.to(device)
    
    rot_mats = util.create_rot_matrices(nbr_rot)
    err = torch.zeros(rot_mats.shape[0])
    
    for i in range(rot_mats.shape[0]):
        a = logsoftmax(y_pred)
        b = logsoftmax(model(input @ rot_mats[i]))
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_err_rand_eta_phi(model, criterion, x, y_pred, y_true, l, nbr_rot, device):
    model.eval()
    # logsoftmax = LogSoftmax(dim=1)
    
    input = x.to(device)
    assert x.shape[2] == 5, "The number of features should be 5"
    
    rot_mats = util.create_rot_matrices(nbr_rot)
    err = torch.zeros(len(rot_mats))
    for i in range(len(rot_mats)):
        # a = logsoftmax(y_pred)
        a = y_pred 
        px_py_pz = input[:, :, :3]
        
        #rot_input = px_py_pz @ rot_mats[i]
        rot_input = px_py_pz @ torch.tensor(rot_mats[i], dtype=torch.float32, device=px_py_pz.device)
        concat_input = torch.cat((rot_input, input[:,:,3:]), dim = 2)
        # b = logsoftmax(model(concat_input))
        b = model(concat_input)
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_error_lorentz(model, criterion, x, y_pred, y_true, l, nbr_rot, device):
    """Function that checks the invariance of the model to Lorentz transformations"""
    model.eval()
    logsoftmax = LogSoftmax(dim=1)
    
    input = x
    rot_mats = util.create_Lorentz_mat(nbr_rot, device)
    rapidity = torch.rand(len(rot_mats))
    err = torch.zeros(len(rot_mats))
    for i in range(len(rot_mats)):
        # any matrix in SO(3,1) can be written as a product of the following matrices
        matrix = rot_mats[i][0] @ torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],[0,0,torch.cosh(rapidity[i]), torch.sinh(rapidity[i])], [0,0,torch.sinh(rapidity[i]), torch.cosh(rapidity[i])]]).to(torch.float32).to(device) @ rot_mats[i][1]
        matrix = torch.transpose(matrix,1,0)
        a = logsoftmax(y_pred)
        b = logsoftmax(model(input @ matrix))
        
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))


def permutation_err(model, criterion, x, y_pred, y_true, l, nbr_rot, device):
    """
    Calculates the invariant error for a given model and input data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        x (torch.Tensor): The input data.
        y_pred (torch.Tensor): The predicted output.
        y_true (torch.Tensor): The true output.
        l (float): The weight for the mean error term.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The total loss, including the mean error and the invariant loss.
    """
    model.eval()
    logsoftmax = LogSoftmax(dim=1)
    
    input = x.to(device)
    err = np.zeros(nbr_rot)
    for i in range(err.shape[0]):
        a = logsoftmax(y_pred)
        b = logsoftmax(model(util.permute_constituents(input).to(device)))
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_err_phi(model, criterion, x, y_pred, y_true, l, nbr_rot, device):
    """
    Calculates the invariant error for a given model by rotating the input data in phi.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        x (torch.Tensor): The input data.
        y_pred (torch.Tensor): The predicted output.
        y_true (torch.Tensor): The true output.
        l (float): The weight for the mean error term.
        nbr_rot (int): The number of rotations to perform. 
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The total loss, including the Cross Entropy loss and the invariant loss.
    """
    model.eval()
    logsoftmax = LogSoftmax(dim=1)
    
    input = x.to(device)
    phi_list = np.linspace(0, 2*np.pi, nbr_rot)
    err = torch.zeros(phi_list.shape[0])
    for i in range(phi_list.shape[0]):
        a = logsoftmax(y_pred)
        phi_dim = input[:, :, 2] + phi_list[i]
        rot_input = torch.cat((input[:, :, 0:2], phi_dim.unsqueeze(-1)), dim = 2)
        b = logsoftmax(model(rot_input))
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))