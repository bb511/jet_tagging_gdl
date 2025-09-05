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

import util
import data.standardization as standardization
import data.rotation_matrices as rm

# apparently one can use torch-geometric.transforms.RandomRotate to rotate the input data in a random way (and it should be efficient)

def create_rot_mat(rot_type, rot_space, nbr_rot):
    if rot_space == 'so3':
        if rot_type == 'scipy':
            rot_mats = rm.create_rot_matrices_scipy(nbr_rot)
        elif rot_type == 'sobol':
            rot_mats = rm.create_rot_mat_sobol(nbr_rot)
        elif rot_type == 'random':
            rot_mats = rm.create_rot_mat_random(nbr_rot)
        elif rot_type == 'regular':
            rot_mats = rm.create_rot_mat_regular(nbr_rot)
        else:
            raise ValueError(f"Unknown rotation type: {rot_type}")
    elif rot_space == 'so2':
        rot_mats = rm.create_so2_rot_matrices(nbr_rot)
    else:
        raise ValueError(f"Unknown rotation space: {rot_space}")
    
    return rot_mats

def inv_err_so2(model, criterion, x, y_pred, y_true, coef_norm , l, nbr_rot, device, compare_to_label=False, return_only_error=False):
    """Supported for px_py_pz and px_py_pz_etarel_phirel data"""
    # model.eval()
    # with torch.no_grad():
    # print("Gradient enabled: ", torch.is_grad_enabled())
    # print("Model training: ", model.training)
    input = x.to(device)
    a = y_pred 
    if coef_norm is not None:
        input = standardization.inverse_standardisation(input, coef_norm)
    
    px_py_pz = input[:, :, :3]
    rot_mats = create_rot_mat(None, 'so2', nbr_rot)
    err = torch.zeros(len(rot_mats))
    for i in range(len(rot_mats)):
        rot_input = px_py_pz @ torch.tensor(rot_mats[i], dtype=torch.float32, device=px_py_pz.device)
        if x.shape[2] == 5:
            concat_input = torch.cat((rot_input, input[:,:,3:]), dim = 2)
        elif x.shape[2] == 3:
            concat_input = rot_input
        else:
            raise ValueError("The number of features should be 3 or 5")
        if coef_norm is not None:
            concat_input = standardization.apply_standardisation(concat_input, coef_norm)
        b = model(concat_input)
        if compare_to_label:
            err[i] = criterion(b, torch.argmax(y_true, dim = 1))
        else:
            err[i] = torch.norm(b-a)/torch.norm(a)
            
    mean_err = err.mean()
    if return_only_error:
        return mean_err
    else:
        return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_err_so3(model, criterion, x, y_pred, y_true, jets, coef_norm , l, rot_type, nbr_rot, device, 
                         filters={"pt":(0,1600), "eta":(-3,3), "etarel":(-0.5,0.5), "phirel":(-0.5,0.5)}, 
                         compare_to_label=False, return_only_error=False):  
    
    """Supported for px_py_pz and px_py_pz_etarel_phirel data"""
    # model.eval()
    # with torch.no_grad():
    # print("Gradient enabled: ", torch.is_grad_enabled())
    # print("Model training: ", model.training)
    input = x.to(device)
    a = y_pred
    px_py_pz = input[:, :, :3]
    jets = jets.unsqueeze(1)  # Shape: [n, 1, feat]
    jets = jets.repeat(1, px_py_pz.shape[1], 1)  # Shape: [n, n_const, feat]
    padded_mask = (px_py_pz[:, :, 0] == 0) & (px_py_pz[:, :, 1] == 0) & (px_py_pz[:, :, 2] == 0)
    if coef_norm is not None:
        input = standardization.inverse_standardisation(input, coef_norm)

    rot_mats = create_rot_mat(rot_type, 'so3', nbr_rot)
    err = torch.zeros(len(rot_mats))
    for i in range(len(rot_mats)):
        R = torch.tensor(rot_mats[i], dtype=torch.float32, device=px_py_pz.device)
        rot_input = px_py_pz @ R
        rot_jets = jets @ R
        
        # Compute eta, phi for rotated input
        pt_data = torch.sqrt(rot_input[:, :, 0]**2 + rot_input[:, :, 1]**2) 
        p_data = torch.sqrt(rot_input[:, :, 0]**2 + rot_input[:, :, 1]**2 + rot_input[:,:,2]**2) 
        
        eta_data = torch.zeros_like(p_data)
        phi_data = torch.zeros_like(p_data)
        non_padded = ~padded_mask
        eps = 1e-6  # Small value to avoid division by zero
        eta_data[non_padded] = 0.5 * torch.log((p_data[non_padded] + rot_input[:,:,2][non_padded]) / (p_data[non_padded] - rot_input[:,:,2][non_padded]+eps))
        phi_data[non_padded] = torch.arctan2(rot_input[:,:,1][non_padded], rot_input[:,:,0][non_padded])

        # Compute eta, phi for rotated jets
        p_jets = torch.sqrt(rot_jets[:, :, 0]**2 + rot_jets[:, :, 1]**2 + rot_jets[:,:,2]**2) 
        eta_jets = 0.5 * torch.log((p_jets + rot_jets[:,:,2]) / (p_jets - rot_jets[:,:,2]+eps))  #DIVISION BY 0? -> INF
        phi_jets = torch.arctan2(rot_jets[:, :, 1], rot_jets[:, :, 0])
        
        # Expand dims for relative computation
        #pt_expanded = pt_data.unsqueeze(2)
        eta_expanded = eta_data.unsqueeze(2)
        phi_expanded = phi_data.unsqueeze(2)
        eta_expanded_jets = eta_jets.unsqueeze(2)
        phi_expanded_jets = phi_jets.unsqueeze(2)

        # Compute relative eta and phi
        rotated_etarel = eta_expanded - eta_expanded_jets
        rotated_phirel = (phi_expanded - phi_expanded_jets+ torch.pi) % (2 * torch.pi) - torch.pi
        rotated_etarel[padded_mask] = 0
        rotated_phirel[padded_mask] = 0
        
        indices_mask = torch.ones(px_py_pz.shape[0], dtype=torch.bool, device=device)
        # Apply filters
        if filters["eta"] is not None:
            eta_min, eta_max = filters["eta"]
            mask = (eta_data >= eta_min) & (eta_data <= eta_max)
            indices_mask &= mask.all(dim=1)

        if filters["pt"] is not None:
            pt_min, pt_max = filters["pt"]
            mask = (pt_data >= pt_min) & (pt_data <= pt_max)
            indices_mask &= mask.all(dim=1)

        if filters["etarel"] is not None:
            etarel_min, etarel_max = filters["etarel"]
            mask = (rotated_etarel[:,:,0] >= etarel_min) & (rotated_etarel[:,:,0] <= etarel_max)
            indices_mask &= mask.all(dim=1)

        if filters["phirel"] is not None:
            phirel_min, phirel_max = filters["phirel"]
            mask = (rotated_phirel[:,:,0] >= phirel_min) & (rotated_phirel[:,:,0] <= phirel_max)
            indices_mask &= mask.all(dim=1)

        rotated_etarel = torch.clamp(rotated_etarel, filters["etarel"][0], filters["etarel"][1])
        rotated_phirel = torch.clamp(rotated_phirel, filters["phirel"][0], filters["phirel"][1])
        
        if x.shape[2] == 5:
            rotated_phirel_etarel = torch.cat((rotated_etarel, rotated_phirel), dim=2)
            concat_input = torch.cat((rot_input, rotated_phirel_etarel), dim = 2)
        elif x.shape[2] == 3:
            concat_input = rot_input
        else:
            raise ValueError("The number of features should be 3 or 5")
        
        # Standardize if needed
        if coef_norm is not None:
            concat_input = standardization.apply_standardisation(concat_input, coef_norm)

        # Forward
        b = model(concat_input)
        #err[i] = torch.norm(b - a) / torch.norm(a)
        # Compute error
        if compare_to_label:
            err[i] = criterion(b, torch.argmax(y_true, dim = 1))
        else:
            if indices_mask.sum() > 0:
                err[i] = torch.norm(b[indices_mask] - a[indices_mask]) / torch.norm(a[indices_mask])
                #print (err[i])
                # print(b, a)
            else:
                err[i] = 0.0
                
    mean_err = err.mean()
    if return_only_error:
        return mean_err
    else:
        return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def check_so3_rot_err(model, x, jets, coef_norm, rot_type, nbr_rot, device,
                  filters={"pt":(0,1600), "eta":(-3,3), "etarel":(-0.5,0.5), "phirel":(-0.5,0.5)},
                  compare_to_label=False):  
    """Function that checks the invariance of the model to rotations for the data
        px_py_pz_eta_rel_phi_rel
        px_py_pz
    """
    model.eval()
    with torch.no_grad():
        input = x
        a = model(input)
        rot_mats = create_rot_mat(rot_type, 'so3', nbr_rot)
        err = np.zeros(len(rot_mats))
        if coef_norm is not None:
            input = standardization.inverse_standardisation(input, coef_norm)
        
        px_py_pz = input[:, :, :3]
        jets = jets.unsqueeze(1)  # Shape: [n, 1, feat]
        jets = jets.repeat(1, px_py_pz.shape[1], 1)  # Shape: [n, n_const, feat]
        padded_mask = (px_py_pz[:, :, 0] == 0) & (px_py_pz[:, :, 1] == 0) & (px_py_pz[:, :, 2] == 0)

        for i in range(len(rot_mats)):
            R = torch.tensor(rot_mats[i], dtype=torch.float32, device=px_py_pz.device)
            rot_input = px_py_pz @ R
            rot_jets = jets @ R
            
            pt_data = torch.sqrt(rot_input[:, :, 0]**2 + rot_input[:, :, 1]**2)
            p_data = torch.sqrt(rot_input[:, :, 0]**2 + rot_input[:, :, 1]**2 + rot_input[:,:,2]**2)
            eta_data = torch.zeros_like(p_data)
            phi_data = torch.zeros_like(p_data)
            non_padded = ~padded_mask
            eta_data[non_padded] = 0.5 * torch.log((p_data[non_padded] + rot_input[:,:,2][non_padded]) / (p_data[non_padded] - rot_input[:,:,2][non_padded]))
            phi_data[non_padded] = torch.arctan2(rot_input[:,:,1][non_padded], rot_input[:,:,0][non_padded])
            
            p_jets = torch.sqrt(rot_jets[:, :, 0]**2 + rot_jets[:, :, 1]**2 + rot_jets[:,:,2]**2)
            eta_jets = 0.5 * torch.log((p_jets + rot_jets[:,:,2]) / (p_jets - rot_jets[:,:,2]))
            phi_jets = torch.arctan2(rot_jets[:, :, 1], rot_jets[:, :, 0])
            
            eta_expanded = eta_data.unsqueeze(2)
            phi_expanded = phi_data.unsqueeze(2)
            eta_expanded_jets = eta_jets.unsqueeze(2)
            phi_expanded_jets = phi_jets.unsqueeze(2)

            rotated_etarel = eta_expanded - eta_expanded_jets
            rotated_phirel = (phi_expanded - phi_expanded_jets+ torch.pi) % (2 * torch.pi) - torch.pi
            rotated_etarel[padded_mask] = 0
            rotated_phirel[padded_mask] = 0
            
            if x.shape[2] == 5:
                rotated_phirel_etarel = torch.cat((rotated_etarel, rotated_phirel), dim=2)
                input_cat = torch.cat((rot_input, rotated_phirel_etarel), dim=2)
            elif x.shape[2] == 3:
                input_cat = rot_input
            else:
                raise ValueError("The number of features should be 3 or 5")
            
            indices_mask = torch.ones(input_cat.shape[0], dtype=torch.bool, device=device)

            # Apply filters
            if filters["eta"] is not None:
                eta_min, eta_max = filters["eta"]
                mask = (eta_data >= eta_min) & (eta_data <= eta_max)
                indices_mask &= mask.all(dim=1)

            if filters["pt"] is not None:
                pt_min, pt_max = filters["pt"]
                mask = (pt_data >= pt_min) & (pt_data <= pt_max)
                indices_mask &= mask.all(dim=1)

            if filters["etarel"] is not None:
                etarel_min, etarel_max = filters["etarel"]
                mask = (rotated_phirel_etarel[:, :, 0] >= etarel_min) & (rotated_phirel_etarel[:, :, 0] <= etarel_max)
                indices_mask &= mask.all(dim=1)

            if filters["phirel"] is not None:
                phirel_min, phirel_max = filters["phirel"]
                mask = (rotated_phirel_etarel[:, :, 1] >= phirel_min) & (rotated_phirel_etarel[:, :, 1] <= phirel_max)
                indices_mask &= mask.all(dim=1)

            if coef_norm is not None:
                input_cat = standardization.apply_standardisation(input_cat, coef_norm)

            b = model(input_cat)
            # err[i] = torch.norm(b - a) / torch.norm(a)
            # Only evaluate error at selected indices
            if indices_mask.sum() > 0:
                err[i] = torch.norm(b[indices_mask] - a[indices_mask]) / torch.norm(a[indices_mask])
            else:
                err[i] = 0.0
 
        mean_err = err.mean()
        return mean_err
    
def check_so2_rot_err(model, x, coef_norm, nbr_rot, device):
    """Supported for px_py_pz and px_py_pz_etarel_phirel data"""
    model.eval()
    with torch.no_grad():
        # print("Gradient enabled: ", torch.is_grad_enabled())
        # print("Model training: ", model.training)
        input = x.to(device)
        a = model(input) 
        if coef_norm is not None:
            input = standardization.inverse_standardisation(input, coef_norm)
        
        px_py_pz = input[:, :, :3]
        rot_mats = create_rot_mat(None, 'so2', nbr_rot)
        err = np.zeros(len(rot_mats))
        for i in range(len(rot_mats)):
            rot_input = px_py_pz @ torch.tensor(rot_mats[i], dtype=torch.float32, device=px_py_pz.device)
            if x.shape[2] == 5:
                concat_input = torch.cat((rot_input, input[:,:,3:]), dim = 2)
            elif x.shape[2] == 3:
                concat_input = rot_input
            else:
                raise ValueError("The number of features should be 3 or 5")
            if coef_norm is not None:
                concat_input = standardization.apply_standardisation(concat_input, coef_norm)
            b = model(concat_input)
            err[i] = torch.norm(b-a)/torch.norm(a)
            
        mean_err = err.mean()
        return mean_err

def permute_constituents(data: torch.Tensor):
    """Applies a random translation to each data point."""
    permutation = torch.randperm(data.size(1))
    data = data[:, permutation]
    return data

def check_permutation_invariance(model, x, nbr_perm, device):
    """Function that checks the permutation invariance of the model"""
    input = x.to(device)
    err = np.zeros(nbr_perm)
    for i in range(err.shape[0]):
        a = model(input)
        b = model(permute_constituents(input).to(device))
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    print("The mean permutation invariance error is ", mean_err * 100, "%")
    return mean_err


###################################################################################################################
###################################################################################################################
###################################################################################################################

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
        b = logsoftmax(model(permute_constituents(input).to(device)))
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))

def inv_error_lorentz(model, criterion, x, y_pred, y_true, l, nbr_rot, device):
    """Function that checks the invariance of the model to Lorentz transformations"""
    model.eval()
    logsoftmax = LogSoftmax(dim=1)
    
    input = x
    rot_mats = rm.create_Lorentz_mat(nbr_rot, device)
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

def check_invariance_Lorentz(model, x, nbr_rot, device):
    """Function that checks the invariance of the model to Lorentz transformations"""
    nbr_const, nbr_feat = x.shape[1], x.shape[2]
    try:
        assert nbr_feat == 4, "The number of features should be 4"
    except:
        print("The number of features should be 4 for the invariance check!")
        return 1000000.0

    input = x.to(device)
    rot_mats = rm.create_Lorentz_mat(nbr_rot, device)
    rapidity = torch.rand(len(rot_mats))
    err = torch.zeros(len(rot_mats))
    for i in range(len(rot_mats)):
        # any matrix in SO(3,1) can be written as a product of the following matrices
        matrix = rot_mats[i][0] @ torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],[0,0,torch.cosh(rapidity[i]), torch.sinh(rapidity[i])], [0,0,torch.sinh(rapidity[i]), torch.cosh(rapidity[i])]]).to(torch.float32).to(device) @ rot_mats[i][1]
        matrix = torch.transpose(matrix,1,0)
        
        a = model(input)
        b = model(input @ matrix)
        
        err[i] = torch.norm(b-a)/torch.norm(a)
    mean_err = err.mean()
    print(f'The mean invariance error is {mean_err*100}% for Lorentz transformations')
    return mean_err*100

# elif x.shape[2] == 4:
        #     rot_mats = create_Lorentz_mat(16, device)
        #     rapidity = torch.rand(len(rot_mats))
        #     err = torch.zeros(len(rot_mats))
        #     for i in range(len(rot_mats)):
        #         matrix = rot_mats[i][0] @ torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
        #                                                [0, 0, torch.cosh(rapidity[i]), torch.sinh(rapidity[i])],
        #                                                [0, 0, torch.sinh(rapidity[i]), torch.cosh(rapidity[i])]]).to(torch.float32).to(device) @ rot_mats[i][1]
        #         matrix = torch.transpose(matrix, 1, 0)
        #         a = model(input)
        #         b = model(input @ matrix)
        #         err[i] = torch.norm(b - a) / torch.norm(a)
        #     mean_err = err.mean()
        #     return mean_err

# def inv_err_so2_eta_phi(model, criterion, x, y_pred, y_true, coef_norm, l, nbr_rot, device):
#     model.eval()
#     input = x.to(device)
#     a = y_pred 
#     px_py_pz = input[:, :, :3]
    
#     if coef_norm is not None:
#         input = standardization.inverse_standardisation(input, coef_norm)
        
#     assert x.shape[2] == 5, "The number of features should be 5"
#     rot_mats = util.create_so2_rot_matrices(nbr_rot)
#     err = torch.zeros(len(rot_mats))
#     for i in range(len(rot_mats)):
#         rot_input = px_py_pz @ torch.tensor(rot_mats[i], dtype=torch.float32, device=px_py_pz.device)
#         concat_input = torch.cat((rot_input, input[:,:,3:]), dim = 2)
        
#         if coef_norm is not None:
#             concat_input = standardization.apply_standardisation(concat_input, coef_norm)
#         b = model(concat_input)
#         err[i] = torch.norm(b-a)/torch.norm(a)
        
#     mean_err = err.mean()
#     return l * mean_err + criterion(y_pred, torch.argmax(y_true, dim = 1))