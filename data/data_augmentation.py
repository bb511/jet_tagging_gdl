import numpy as np
from tqdm import tqdm
import data.rotation_matrices as rm

class DataAugmentation:
    def __init__(self, data_type="px_py_pz_etarel_phirel", augmentation_type='scipy', augmentation_space='so3', num_rotations=8, seed=None):
        self.data_type = data_type
        self.augmentation_type = augmentation_type
        self.augmentation_space = augmentation_space
        self.num_rotations = num_rotations
        self.seed = seed
        
        if self.data_type not in ['px_py_pz', 'px_py_pz_etarel_phirel']:
            raise ValueError(f"Data type {self.data_type} is not supported. At the moment, supported types are: 'px_py_pz', 'px_py_pz_etarel_phirel'.")
        
        if self.augmentation_space=='so3':
            if self.augmentation_type == 'sobol':
                self.rotation_matrices = rm.create_rot_mat_sobol(self.num_rotations)
            elif self.augmentation_type == 'scipy':
                if self.seed is None:
                    self.rotation_matrices = rm.create_rot_matrices_scipy(self.num_rotations)
                else:
                    self.rotation_matrices = rm.create_rot_matrices_scipy(self.num_rotations, seed=self.seed)
            elif self.augmentation_type == 'random':
                if self.seed is None:
                    self.rotation_matrices = rm.create_rot_mat_random(self.num_rotations)
                else:
                    self.rotation_matrices = rm.create_rot_mat_random(self.num_rotations, seed=self.seed)
            elif self.augmentation_type == 'regular':
                self.rotation_matrices = rm.create_rot_mat_regular(self.num_rotations)
            else:
                raise ValueError(f"Unknown augmentation type: {self.augmentation_type}")
        elif self.augmentation_space=='so2':
            if self.seed is None:
                self.rotation_matrices = rm.create_so2_rot_matrices(self.num_rotations)
            else:
                self.rotation_matrices = rm.create_so2_rot_matrices(self.num_rotations, seed=self.seed)
        else:
            raise ValueError(f"Unknown augmentation space: {self.augmentation_space}")

    def px_py_pz_to_pt_eta_phi(self, data):
        px = data[:, :, 0]
        py = data[:, :, 1]
        pz = data[:, :, 2]
        
        padded_mask = (px == 0) & (py == 0) & (pz == 0)

        p_T = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)

        # Initialize outputs as zeros
        eta = np.zeros_like(p_T)
        phi = np.zeros_like(p_T)

        # Only compute for non-padded constituents
        non_padded = ~padded_mask
        eta[non_padded] = 0.5 * np.log((p[non_padded] + pz[non_padded]) / (p[non_padded] - pz[non_padded]))
        phi[non_padded] = np.arctan2(py[non_padded], px[non_padded])

        # Expand dims for later concatenation
        p_T = np.expand_dims(p_T, axis=2)
        eta = np.expand_dims(eta, axis=2)
        phi = np.expand_dims(phi, axis=2)

        return p_T, eta, phi
    
    def augment_data_so3(self, data, labels, jets, filters={"pt": None, "eta": None, "etarel": None, "phirel": None}, shuffle=True):
        if self.data_type == 'px_py_pz':
            data_px_py_pz = data[:, :, :3]
            data_phirel_etarel = None
        elif self.data_type =='px_py_pz_etarel_phirel':
            data_px_py_pz = data[:, :, :3]
            data_phirel_etarel = data[:, :, -2:]
            
        augmented_data = [data]
        augmented_labels = [labels]
        augmented_jets = [jets]
        
        const = data.shape[1]  # Number of constituents (from your main data array)
        jets = np.expand_dims(jets, axis=1)         # Shape: [n, 1, feat]
        jets = np.repeat(jets, const, axis=1)  # Shape: [n, const, feat]
        
        for i, rotation_matrix in enumerate(tqdm(self.rotation_matrices, desc="Augmenting data")):
            rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
            new_labels = labels.copy()
            
            pt_expanded, eta_expanded, phi_expanded = self.px_py_pz_to_pt_eta_phi(rotated_data)
            rotated_jets = np.dot(jets, rotation_matrix.T)
            _, eta_expanded_jets, phi_expanded_jets = self.px_py_pz_to_pt_eta_phi(rotated_jets)
            rotated_etarel = (eta_expanded - eta_expanded_jets) #* np.sign(eta_expanded_jets)
            rotated_phirel = (phi_expanded - phi_expanded_jets + np.pi) % (2 * np.pi) - np.pi
            padded_mask = (rotated_data[:, :, 0] == 0) & (rotated_data[:, :, 1] == 0) & (rotated_data[:, :, 2] == 0)
            rotated_etarel[padded_mask] = 0
            rotated_phirel[padded_mask] = 0
            rotated_phirel_etarel = np.concatenate((rotated_etarel, rotated_phirel), axis=2)
                
            if filters["eta"] is not None:
                eta_min , eta_max = filters["eta"]  
                mask = (eta_expanded >= eta_min) & (eta_expanded <= eta_max)
                jets_to_keep = np.all(mask, axis=1).squeeze() 
                rotated_data = rotated_data[jets_to_keep]
                eta_expanded = eta_expanded[jets_to_keep]
                rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
                new_labels = new_labels[jets_to_keep]
                pt_expanded = pt_expanded[jets_to_keep]
                rotated_jets = rotated_jets[jets_to_keep]
                    
            if filters["pt"] is not None:
                pt_min , pt_max = filters["pt"]  
                mask = (pt_expanded >= pt_min) & (pt_expanded <= pt_max)
                jets_to_keep = np.all(mask, axis=1).squeeze()   
                rotated_data = rotated_data[jets_to_keep]
                pt_expanded = pt_expanded[jets_to_keep]
                rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
                new_labels = new_labels[jets_to_keep]
                eta_expanded = eta_expanded[jets_to_keep]
                rotated_jets = rotated_jets[jets_to_keep]
                    
            if filters["etarel"] is not None:
                etarel_min, etarel_max = filters["etarel"]
                mask = (rotated_phirel_etarel[:, :, 0] >= etarel_min) & (rotated_phirel_etarel[:, :, 0] <= etarel_max)
                jets_to_keep = np.all(mask, axis=1).squeeze()
                rotated_data = rotated_data[jets_to_keep]
                eta_expanded = eta_expanded[jets_to_keep]
                rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
                new_labels = new_labels[jets_to_keep]
                pt_expanded = pt_expanded[jets_to_keep]
                rotated_jets = rotated_jets[jets_to_keep]
                
            if filters["phirel"] is not None:
                phirel_min, phirel_max = filters["phirel"]
                mask = (rotated_phirel_etarel[:, :, 1] >= phirel_min) & (rotated_phirel_etarel[:, :, 1] <= phirel_max)
                jets_to_keep = np.all(mask, axis=1).squeeze()
                rotated_data = rotated_data[jets_to_keep]
                eta_expanded = eta_expanded[jets_to_keep]
                rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
                new_labels = new_labels[jets_to_keep]
                pt_expanded = pt_expanded[jets_to_keep]
                rotated_jets = rotated_jets[jets_to_keep]
                
            if self.data_type == 'px_py_pz':
                new_data = rotated_data
            elif self.data_type == 'px_py_pz_etarel_phirel':
                new_data = np.concatenate((rotated_data, rotated_phirel_etarel), axis=2)
            
            augmented_data.append(new_data)
            augmented_labels.append(new_labels)
            augmented_jets.append(rotated_jets[:,0,:])
                    
        augmented_data = np.concatenate(augmented_data, axis=0)
        augmented_labels = np.concatenate(augmented_labels, axis=0)
        augmented_jets = np.concatenate(augmented_jets, axis=0)
        
        if shuffle:
            indices = np.arange(augmented_data.shape[0])
            np.random.shuffle(indices)
            augmented_data = augmented_data[indices]
            augmented_labels = augmented_labels[indices]
            augmented_jets = augmented_jets[indices]

        return augmented_data, augmented_labels, augmented_jets
    
    def augment_data_so2(self, data, labels, jets, shuffle=True):
        if self.data_type == 'px_py_pz':
            data_px_py_pz = data[:, :, :3]
            data_phirel_etarel = None
        elif self.data_type =='px_py_pz_etarel_phirel':
            data_px_py_pz = data[:, :, :3]
            data_phirel_etarel = data[:, :, -2:]
        else:
            raise ValueError(f"Not supported data type: {self.data_type}")
            
        augmented_data = [data]
        augmented_labels = [labels]
        augmented_jets = [jets]
        const = data.shape[1]
        
        for i, rotation_matrix in enumerate(tqdm(self.rotation_matrices, desc="Augmenting data")):
            rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
            new_labels = labels.copy()
            rotated_jets = np.dot(jets, rotation_matrix.T)
            
            if self.data_type == 'px_py_pz':
                new_data = rotated_data
            elif self.data_type == 'px_py_pz_etarel_phirel':
                new_data = np.concatenate((rotated_data, data_phirel_etarel), axis=2)
            augmented_data.append(new_data)
            augmented_labels.append(new_labels)
            augmented_jets.append(rotated_jets)
                
        augmented_data = np.concatenate(augmented_data, axis=0)
        augmented_labels = np.concatenate(augmented_labels, axis=0)
        augmented_jets = np.concatenate(augmented_jets, axis=0)
        if shuffle:
            indices = np.arange(augmented_data.shape[0])
            np.random.shuffle(indices)
            augmented_data = augmented_data[indices]
            augmented_labels = augmented_labels[indices]
            augmented_jets = augmented_jets[indices]
            
        return augmented_data, augmented_labels, augmented_jets
    
    def augment_data(self, data, labels, jets, filters={"pt": None, "eta": None, "etarel": None, "phirel": None}, shuffle=True):
        if self.augmentation_space=='so3':
            return self.augment_data_so3(data, labels, jets, filters=filters, shuffle=shuffle)
        elif self.augmentation_space=='so2':
            return self.augment_data_so2(data, labels, jets, shuffle=shuffle)

# def augment_data(data, labels, jets, rotation_matrices, filters = {"pt": None, "eta": None, "etarel": None, "phirel": None}, data_type='px_py_pz_etarel_phirel'):
    
#     data_px_py_pz = data[:, :, :3] 
#     data_phirel_etarel = data[:, :, -2:]
    
#     augmented_data = [data]
#     augmented_labels = [labels]
#     augmented_jets = [jets]
    
#     const = data.shape[1]  # Number of constituents (from your main data array)
#     jets = np.expand_dims(jets, axis=1)         # Shape: [n, 1, feat]
#     jets = np.repeat(jets, const, axis=1)  # Shape: [n, const, feat]
    
#     for i, rotation_matrix in enumerate(tqdm(rotation_matrices, desc="Augmenting data")):
#         rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
#         pt_expanded, eta_expanded, phi_expanded = calculate_features(rotated_data)
#         rotated_jets = np.dot(jets, rotation_matrix.T)
        
#         if 'etarelinv_phirelinv' in data_type:
#             rotated_phirel_etarel = data_phirel_etarel
#         elif 'etarel_phirel' in data_type:
#             _, eta_expanded_jets, phi_expanded_jets = calculate_features(rotated_jets)
#             rotated_etarel = (eta_expanded - eta_expanded_jets) #* np.sign(eta_expanded_jets)
#             rotated_phirel = (phi_expanded - phi_expanded_jets + np.pi) % (2 * np.pi) - np.pi
            
#             padded_mask = (rotated_data[:, :, 0] == 0) & (rotated_data[:, :, 1] == 0) & (rotated_data[:, :, 2] == 0)
#             rotated_etarel[padded_mask] = 0
#             rotated_phirel[padded_mask] = 0
            
#             rotated_phirel_etarel = np.concatenate((rotated_etarel, rotated_phirel), axis=2)
            
#         new_labels = labels.copy()
            
#         if filters["eta"] is not None:
#             eta_min , eta_max = filters["eta"]  
#             mask = (eta_expanded >= eta_min) & (eta_expanded <= eta_max)
#             jets_to_keep = np.all(mask, axis=1).squeeze() 
#             rotated_data = rotated_data[jets_to_keep]
#             eta_expanded = eta_expanded[jets_to_keep]
#             rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#             new_labels = new_labels[jets_to_keep]
#             pt_expanded = pt_expanded[jets_to_keep]
#             rotated_jets = rotated_jets[jets_to_keep]
                
#         if filters["pt"] is not None:
#             pt_min , pt_max = filters["pt"]  
#             mask = (pt_expanded >= pt_min) & (pt_expanded <= pt_max)
#             jets_to_keep = np.all(mask, axis=1).squeeze()   
#             rotated_data = rotated_data[jets_to_keep]
#             pt_expanded = pt_expanded[jets_to_keep]
#             rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#             new_labels = new_labels[jets_to_keep]
#             eta_expanded = eta_expanded[jets_to_keep]
#             rotated_jets = rotated_jets[jets_to_keep]
                
#         if filters["etarel"] is not None:
#             etarel_min, etarel_max = filters["etarel"]
#             mask = (rotated_phirel_etarel[:, :, 0] >= etarel_min) & (rotated_phirel_etarel[:, :, 0] <= etarel_max)
#             jets_to_keep = np.all(mask, axis=1).squeeze()
#             rotated_data = rotated_data[jets_to_keep]
#             eta_expanded = eta_expanded[jets_to_keep]
#             rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#             new_labels = new_labels[jets_to_keep]
#             pt_expanded = pt_expanded[jets_to_keep]
#             rotated_jets = rotated_jets[jets_to_keep]
            
#         if filters["phirel"] is not None:
#             phirel_min, phirel_max = filters["phirel"]
#             mask = (rotated_phirel_etarel[:, :, 1] >= phirel_min) & (rotated_phirel_etarel[:, :, 1] <= phirel_max)
#             jets_to_keep = np.all(mask, axis=1).squeeze()
#             rotated_data = rotated_data[jets_to_keep]
#             eta_expanded = eta_expanded[jets_to_keep]
#             rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#             new_labels = new_labels[jets_to_keep]
#             pt_expanded = pt_expanded[jets_to_keep]
#             rotated_jets = rotated_jets[jets_to_keep]
            
#         if data_type == 'px_py_pz':
#             new_data = rotated_data
#         elif data_type == 'px_py_pz_etarel_phirel' or data_type == 'px_py_pz_etarelinv_phirelinv':
#             new_data = np.concatenate((rotated_data, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_eta_etarel_phirel':
#             new_data = np.concatenate((rotated_data, eta_expanded, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_pt_etarel_phirel':
#             new_data = np.concatenate((rotated_data, pt_expanded, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_pt_eta_etarel_phirel':
#             new_data = np.concatenate((rotated_data, pt_expanded, eta_expanded, rotated_phirel_etarel), axis=2)
        
#         augmented_data.append(new_data)
#         augmented_labels.append(new_labels)
#         augmented_jets.append(rotated_jets[:,0,:])
                
#     augmented_data = np.concatenate(augmented_data, axis=0)
#     augmented_labels = np.concatenate(augmented_labels, axis=0)
#     augmented_jets = np.concatenate(augmented_jets, axis=0)
    
#     # Shuffle in unison
#     indices = np.arange(augmented_data.shape[0])
#     np.random.shuffle(indices)
#     augmented_data = augmented_data[indices]
#     augmented_labels = augmented_labels[indices]
#     augmented_jets = augmented_jets[indices]

#     return augmented_data, augmented_labels, augmented_jets

# def augment_data_fixed_single(data, labels, jets, rotation_matrices, filters={"pt": None, "eta": None}, data_type='px_py_pz_etarel_phirel'):
#     data_px_py_pz = data[:, :, :3]
#     data_phirel_etarel = data[:, :, -2:]
#     augmented_data = [data]
#     augmented_labels = [labels]
#     const = data.shape[1]

#     for rotation_matrix in rotation_matrices:
#         rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
#         pt_exp, eta_exp, phi_exp = calculate_features(rotated_data)

#         # Correct: rotate jets only once
#         rotated_jets = np.dot(jets, rotation_matrix.T)
#         _, eta_jets, phi_jets = calculate_features(rotated_jets[:, None, :])  # shape: (N, 1, 1)
#         eta_jets = np.repeat(eta_jets, const, axis=1)  # shape: (N, C, 1)
#         phi_jets = np.repeat(phi_jets, const, axis=1)  # shape: (N, C, 1)

#         if 'etarelinv_phirelinv' in data_type:
#             rotated_phirel_etarel = data_phirel_etarel
#         elif 'etarel_phirel' in data_type:
#             rotated_etarel = eta_exp - eta_jets
#             rotated_phirel = phi_exp - phi_jets
#             rotated_phirel = (rotated_phirel + np.pi) % (2 * np.pi) - np.pi
#             rotated_phirel_etarel = np.concatenate((rotated_etarel, rotated_phirel), axis=2)

#         new_labels = labels.copy()

#         if filters["eta"] is not None:
#             eta_min, eta_max = filters["eta"]
#             mask = (eta_exp >= eta_min) & (eta_exp <= eta_max)
#             jets_to_keep = np.all(mask, axis=1).squeeze()
#             rotated_data = rotated_data[jets_to_keep]
#             eta_exp = eta_exp[jets_to_keep]
#             rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#             new_labels = new_labels[jets_to_keep]
#             pt_exp = pt_exp[jets_to_keep]

#         if filters["pt"] is not None:
#             pt_min, pt_max = filters["pt"]
#             mask = (pt_exp >= pt_min) & (pt_exp <= pt_max)
#             jets_to_keep = np.all(mask, axis=1).squeeze()
#             rotated_data = rotated_data[jets_to_keep]
#             pt_exp = pt_exp[jets_to_keep]
#             rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#             new_labels = new_labels[jets_to_keep]
#             eta_exp = eta_exp[jets_to_keep]

#         if data_type == 'px_py_pz':
#             new_data = rotated_data
#         elif data_type in ['px_py_pz_etarel_phirel', 'px_py_pz_etarelinv_phirelinv']:
#             new_data = np.concatenate((rotated_data, rotated_phirel_etarel), axis=2)

#         augmented_data.append(new_data)
#         augmented_labels.append(new_labels)

#     augmented_data = np.concatenate(augmented_data, axis=0)
#     augmented_labels = np.concatenate(augmented_labels, axis=0)
#     indices = np.arange(augmented_data.shape[0])
#     return augmented_data[indices], augmented_labels[indices]
    
# def calculate_features(data):
#     # px = data[:, :, 0]
#     # py = data[:, :, 1]
#     # pz = data[:, :, 2]
#     p_T = np.sqrt(data[:, :, 0]**2 + data[:, :, 1]**2)
#     epsilon = 1e-10
#     p_T = np.maximum(p_T, epsilon)
#     theta = np.arctan2(p_T, data[:, :, 2])
#     eta = -np.log(np.tan(0.5 * theta))
#     phi = np.arctan2(data[:, :, 1], data[:, :, 0])
    
#     p_T = np.expand_dims(p_T, axis=2)
#     eta = np.expand_dims(eta, axis=2)
#     phi = np.expand_dims(phi, axis=2)
    
#     return p_T, eta, phi

# def augment_data(data, labels, jets, rotation_matrices, filters = {"pt": None, "eta": None}, data_type='px_py_pz_etarel_phirel'):
    
#     data_px_py_pz = data[:, :, :3] 
#     data_phirel_etarel = data[:, :, -2:]
    
#     augmented_data = [data]
#     augmented_labels = [labels]
    
#     const = data.shape[1]  # Number of constituents (from your main data array)
#     jets = np.expand_dims(jets, axis=1)         # Shape: [n, 1, feat]
#     jets = np.repeat(jets, const, axis=1)  # Shape: [n, const, feat]
    
#     for i, rotation_matrix in enumerate(tqdm(rotation_matrices, desc="Augmenting data")):
#         rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
#         pt_expanded, eta_expanded, phi_expanded = calculate_features(rotated_data)
        
#         if 'etarelinv_phirelinv' in data_type:
#             rotated_phirel_etarel = data_phirel_etarel
#         elif 'etarel_phirel' in data_type:
#             rotated_jets = np.dot(jets, rotation_matrix.T)
#             pt_expanded_jets, eta_expanded_jets, phi_expanded_jets = calculate_features(rotated_jets)
#             rotated_etarel = eta_expanded - eta_expanded_jets
#             rotated_phirel = phi_expanded - phi_expanded_jets
#             rotated_phirel_etarel = np.concatenate((rotated_etarel, rotated_phirel), axis=2)
            
#         new_labels = labels.copy()
            
#         if filters["eta"] is not None:
#                 eta_min , eta_max = filters["eta"]  
#                 mask = (eta_expanded >= eta_min) & (eta_expanded <= eta_max)
#                 jets_to_keep = np.all(mask, axis=1).squeeze() 
#                 rotated_data = rotated_data[jets_to_keep]
#                 eta_expanded = eta_expanded[jets_to_keep]
#                 rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#                 new_labels = new_labels[jets_to_keep]
#                 pt_expanded = pt_expanded[jets_to_keep]
                
#         if filters["pt"] is not None:
#                 pt_min , pt_max = filters["pt"]  
#                 mask = (pt_expanded >= pt_min) & (pt_expanded <= pt_max)
#                 jets_to_keep = np.all(mask, axis=1).squeeze()   
#                 rotated_data = rotated_data[jets_to_keep]
#                 pt_expanded = pt_expanded[jets_to_keep]
#                 rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#                 new_labels = new_labels[jets_to_keep]
#                 eta_expanded = eta_expanded[jets_to_keep]
        
#         if data_type == 'px_py_pz':
#             new_data = rotated_data
#         elif data_type == 'px_py_pz_etarel_phirel' or data_type == 'px_py_pz_etarelinv_phirelinv':
#             new_data = np.concatenate((rotated_data, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_eta_etarel_phirel':
#             new_data = np.concatenate((rotated_data, eta_expanded, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_pt_etarel_phirel':
#             new_data = np.concatenate((rotated_data, pt_expanded, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_pt_eta_etarel_phirel':
#             new_data = np.concatenate((rotated_data, pt_expanded, eta_expanded, rotated_phirel_etarel), axis=2)
        
#         augmented_data.append(new_data)
#         augmented_labels.append(new_labels)
                
#     augmented_data = np.concatenate(augmented_data, axis=0)
#     augmented_labels = np.concatenate(augmented_labels, axis=0)
    
#     # Shuffle in unison
#     indices = np.arange(augmented_data.shape[0])
#     np.random.shuffle(indices)
#     augmented_data = augmented_data[indices]
#     augmented_labels = augmented_labels[indices]

#     return augmented_data, augmented_labels

# def eta_from_log(p, eps=1e-10):
#     """
#     Compute pseudorapidity η = ½ ln[(|p| + pz)/(|p| - pz)] for a 3-vector p = (px, py, pz).
#     Supports scalar or array inputs.
#     """
#     p = np.asarray(p)
#     # norm of the 3-momentum
#     p_norm = np.linalg.norm(p, axis=-1)
#     pz = p[..., 2]
#     # avoid zeros in denominator
#     num = p_norm + pz
#     den = p_norm - pz + eps
#     return 0.5 * np.log(num / den)

# def calculate_features(data):
#     px = data[:, :, 0]
#     py = data[:, :, 1]
#     pz = data[:, :, 2]

#     p_T = np.sqrt(px**2 + py**2)
#     p = np.sqrt(px**2 + py**2 + pz**2)
#     epsilon = 1e-10
#     #p = np.maximum(p, epsilon)

#     eta = 0.5 * np.log((p + pz) / (p - pz + epsilon))
#     phi = np.arctan2(py, px)

#     # Expand dims for later concatenation
#     p_T = np.expand_dims(p_T, axis=2)
#     eta = np.expand_dims(eta, axis=2)
#     phi = np.expand_dims(phi, axis=2)

#     return p_T, eta, phi


# def augment_data(data, labels, jets, rotation_matrices, filters = {"pt": None, "eta": None}, data_type='px_py_pz_etarel_phirel'):
    
#     data_px_py_pz = data[:, :, :3] 
#     data_phirel_etarel = data[:, :, -2:]
    
#     augmented_data = [data]
#     augmented_labels = [labels]
    
#     const = data.shape[1]  # Number of constituents (from your main data array)
#     jets = np.expand_dims(jets, axis=1)         # Shape: [n, 1, feat]
#     jets = np.repeat(jets, const, axis=1)  # Shape: [n, const, feat]
    
#     for i, rotation_matrix in enumerate(tqdm(rotation_matrices, desc="Augmenting data")):
#         rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
#         pt_expanded, eta_expanded, phi_expanded = calculate_features(rotated_data)
        
#         if 'etarelinv_phirelinv' in data_type:
#             rotated_phirel_etarel = data_phirel_etarel
#         elif 'etarel_phirel' in data_type:
#             rotated_jets = np.dot(jets, rotation_matrix.T)
#             pt_expanded_jets, eta_expanded_jets, phi_expanded_jets = calculate_features(rotated_jets)
#             rotated_etarel = (eta_expanded - eta_expanded_jets) #* np.sign(eta_expanded_jets)
#             rotated_phirel = (phi_expanded - phi_expanded_jets + np.pi) % (2 * np.pi) - np.pi

#             rotated_phirel_etarel = np.concatenate((rotated_etarel, rotated_phirel), axis=2)
            
#         new_labels = labels.copy()
            
#         if filters["eta"] is not None:
#                 eta_min , eta_max = filters["eta"]  
#                 mask = (eta_expanded >= eta_min) & (eta_expanded <= eta_max)
#                 jets_to_keep = np.all(mask, axis=1).squeeze() 
#                 rotated_data = rotated_data[jets_to_keep]
#                 eta_expanded = eta_expanded[jets_to_keep]
#                 rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#                 new_labels = new_labels[jets_to_keep]
#                 pt_expanded = pt_expanded[jets_to_keep]
                
#         if filters["pt"] is not None:
#                 pt_min , pt_max = filters["pt"]  
#                 mask = (pt_expanded >= pt_min) & (pt_expanded <= pt_max)
#                 jets_to_keep = np.all(mask, axis=1).squeeze()   
#                 rotated_data = rotated_data[jets_to_keep]
#                 pt_expanded = pt_expanded[jets_to_keep]
#                 rotated_phirel_etarel = rotated_phirel_etarel[jets_to_keep]
#                 new_labels = new_labels[jets_to_keep]
#                 eta_expanded = eta_expanded[jets_to_keep]
        
#         if data_type == 'px_py_pz':
#             new_data = rotated_data
#         elif data_type == 'px_py_pz_etarel_phirel' or data_type == 'px_py_pz_etarelinv_phirelinv':
#             new_data = np.concatenate((rotated_data, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_eta_etarel_phirel':
#             new_data = np.concatenate((rotated_data, eta_expanded, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_pt_etarel_phirel':
#             new_data = np.concatenate((rotated_data, pt_expanded, rotated_phirel_etarel), axis=2)
#         elif data_type == 'px_py_pz_pt_eta_etarel_phirel':
#             new_data = np.concatenate((rotated_data, pt_expanded, eta_expanded, rotated_phirel_etarel), axis=2)
        
#         augmented_data.append(new_data)
#         augmented_labels.append(new_labels)
                
#     augmented_data = np.concatenate(augmented_data, axis=0)
#     augmented_labels = np.concatenate(augmented_labels, axis=0)
    
#     # Shuffle in unison
#     # indices = np.arange(augmented_data.shape[0])
#     # np.random.shuffle(indices)
#     # augmented_data = augmented_data[indices]
#     # augmented_labels = augmented_labels[indices]

#     return augmented_data, augmented_labels

# def main(feature_files, label_files, num_rotations, sampling_method):
#     path = "/mnt/c/users/bonvi/Documents/deepset/data/"#'/work/mbonvin/data/'
#     output_dir = os.path.join(path, 'augmented_data')
   
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     if sampling_method == 'sobol':
#         rotation_matrices = create_rot_mat_sobol(num_rotations)
#     elif sampling_method == 'random':
#         rotation_matrices = create_rot_mat_random(num_rotations)
#     elif sampling_method == 'regular':
#         rotation_matrices = create_rot_mat_regular(num_rotations)
#     else:
#         raise ValueError("Invalid sampling method. Choose 'sobol', 'random', or 'regular'.")

#     for feature_file, label_file in zip(feature_files, label_files):
#         input_feature_file = os.path.join(path, feature_file)
#         input_label_file = os.path.join(path, label_file)

#         # Open input files as memory-mapped arrays
#         features = np.load(input_feature_file, mmap_mode='r')
#         labels = np.load(input_label_file, mmap_mode='r')
        
#         print(f"Processing {feature_file}")
#         print(f"Loaded feature data shape: {features.shape}")
#         print(f"Loaded label data shape: {labels.shape}")

#         # Create memory-mapped output files
#         output_feature_file = os.path.join(output_dir, f'augmented_features_{sampling_method}_{num_rotations}.npy')
#         output_label_file = os.path.join(output_dir, f'augmented_labels_{sampling_method}_{num_rotations}.npy')
        
#         augmented_features_shape = (features.shape[0] * num_rotations, features.shape[1], features.shape[2])
#         augmented_labels_shape = (labels.shape[0] * num_rotations, labels.shape[1])
        
#         augmented_features = np.lib.format.open_memmap(output_feature_file, mode='w+', dtype=features.dtype, shape=augmented_features_shape)
#         augmented_labels = np.lib.format.open_memmap(output_label_file, mode='w+', dtype=labels.dtype, shape=augmented_labels_shape)

#         # Process all data at once
#         augmented_features[:], augmented_labels[:] = augment_data(features, labels, rotation_matrices)

#         print(f"Augmentation complete. Augmented features saved to {output_feature_file}")
#         print(f"Augmented labels saved to {output_label_file}")
#         print(f"Final augmented feature shape: {augmented_features.shape}")
#         print(f"Final augmented label shape: {augmented_labels.shape}")

# def old_augment_data(data, labels, rotation_matrices):
#     data_px_py_pz = data[:, :, :3] 
#     data_pt_eta = data[:, :, 3:]    
    
#     augmented_data = np.empty((data.shape[0] * len(rotation_matrices), data.shape[1], data.shape[2]), dtype=data.dtype)
#     augmented_labels = np.empty((labels.shape[0] * len(rotation_matrices), labels.shape[1]), dtype=labels.dtype)
    
#     for i, rotation_matrix in enumerate(tqdm(rotation_matrices, desc="Augmenting data")):
#         rotated_data = np.dot(data_px_py_pz, rotation_matrix.T)
        
#         augmented_data[i::len(rotation_matrices)] = np.concatenate((rotated_data, data_pt_eta), axis=2)
#         augmented_labels[i::len(rotation_matrices)] = labels
    
#     return augmented_data, augmented_labels


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Memory-mapped data augmentation with different SO(3) sampling methods')
#     parser.add_argument('--sampling_method', type=str, choices=['sobol', 'random', 'regular'], default='sobol',
#                         help='Sampling method for rotation matrices (default: sobol)')
#     parser.add_argument('--num_rotations', type=int, default=8,
#                         help='Number of rotations to generate (default: 8)')
#     args = parser.parse_args()

#     feature_files = [
#         #'jetid_px_py_pz_eta_rel_phi_rel/processed/x_train_robust_8const_ptetaphi.npy',
#         #'jetid_px_py_pz_eta_rel_phi_rel_16/processed/x_train_robust_16const_ptetaphi.npy',
#         'jetid_px_py_pz_eta_rel_phi_rel_32/processed/x_train_robust_32const_ptetaphi.npy'
#     ]
#     label_files = [
#         #'jetid_px_py_pz_eta_rel_phi_rel/processed/y_train_robust_8const_ptetaphi.npy',
#         #'jetid_px_py_pz_eta_rel_phi_rel_16/processed/y_train_robust_16const_ptetaphi.npy',
#         'jetid_px_py_pz_eta_rel_phi_rel_32/processed/y_train_robust_32const_ptetaphi.npy'
#     ]
    
#     main(feature_files, label_files, args.num_rotations, args.sampling_method)