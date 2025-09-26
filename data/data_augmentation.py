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
        
        if self.data_type not in ['px_py_pz', 'px_py_pz_etarel_phirel', 'px_py_pz_E', 'px_py_pz_E_etarel_phirel']:
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
        elif self.augmentation_space=='lorentz':
            if self.seed is None:
                self.rotation_matrices = rm.create_lorentz_so2_rot_boost(self.num_rotations)
            else:
                self.rotation_matrices = rm.create_lorentz_so2_rot_boost(self.num_rotations, seed=self.seed)
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
            else:
                raise ValueError(f"Not supported data type: {self.data_type}")
            
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
            else:
                raise ValueError(f"Not supported data type: {self.data_type}")
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
    
    def augment_data_lorentz(self, data, labels, jets, shuffle=True):
        if self.data_type == 'px_py_pz_E':
            data_px_py_pz = data[:, :, :4]
            data_phirel_etarel = None
        elif self.data_type =='px_py_pz_E_etarel_phirel':
            data_px_py_pz = data[:, :, :4]
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
            
            if self.data_type == 'px_py_pz_E':
                new_data = rotated_data
            elif self.data_type == 'px_py_pz_E_etarel_phirel':
                new_data = np.concatenate((rotated_data, data_phirel_etarel), axis=2)
            else:
                raise ValueError(f"Not supported data type: {self.data_type}")
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
        elif self.augmentation_space=='lorentz':
            return self.augment_data_lorentz(data, labels, jets, shuffle=shuffle)

