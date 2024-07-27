# This code is adapted from https://github.com/jet-universe/particle_transformer/blob/main/get_datasets.py
import numpy as np
import awkward as ak
import uproot
import vector
import argparse
import os
vector.register_awkward()

def sort_by_feature(data, feature_index=4, descending=True):
        """
        Sort the data and labels based on a specific feature.

        Args:
            data (np.ndarray): The input data array.
            labels (np.ndarray): The corresponding labels.
            feature_index (int): The index of the feature to sort by.
            descending (bool): Whether to sort in descending order.

        Returns:
            tuple: Sorted data, labels and jet_data.
        """
        sort_indices = np.argsort(data[:, feature_index, :], axis=1)
        if descending:
            sort_indices = np.flip(sort_indices, axis=1)
            
        sample_indices = np.arange(data.shape[0])[:, np.newaxis, np.newaxis]
        feature_indices = np.arange(data.shape[1])[np.newaxis, :, np.newaxis]
        expanded_sorting_indices = sort_indices[:, np.newaxis, :]
        return data[sample_indices, feature_indices, expanded_sorting_indices]

def restrict_nb_constituents(data, nconst):
    """
    Restrict the number of constituents in each jet to a fixed number.

    Args:
        x_data (np.ndarray): The input data array.
        nconst (int): The number of constituents to keep.

    Returns:
        np.ndarray: Data array with restricted number of constituents.
    """
    data = np.pad(data[:, :, :nconst], ((0, 0), (0, 0), (0, max(0, nconst - data.shape[2]))))
    return data
def read_file(
        dirpath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    """Loads a single file from the JetClass dataset.

    **Arguments**

    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet. 
        Jets with fewer particles will be zero-padded, 
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded. 
        The available particle-level features are:
            - part_px
            - part_py
            - part_pz
            - part_energy
            - part_pt
            - part_eta
            - part_phi
            - part_deta: np.where(jet_eta>0, part_eta-jet_p4, -(part_eta-jet_p4))
            - part_dphi: delta_phi(part_phi, jet_phi)
            - part_d0val
            - part_d0err
            - part_dzval
            - part_dzerr
            - part_charge
            - part_isChargedHadron
            - part_isNeutralHadron
            - part_isPhoton
            - part_isElectron
            - part_isMuon
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded. 
        The available jet-level features are:
            - jet_pt
            - jet_eta
            - jet_phi
            - jet_energy
            - jet_nparticles
            - jet_sdmass
            - jet_tau1
            - jet_tau2
            - jet_tau3
            - jet_tau4
    - **labels** : _List[str]_
        - A list of truth labels to be loaded. 
        The available label names are:
            - label_QCD
            - label_Hbb
            - label_Hcc
            - label_Hgg
            - label_H4q
            - label_Hqql
            - label_Zqq
            - label_Wqq
            - label_Tbqq
            - label_Tbl

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features 
                         in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
                    in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth lables
               in the shape `(num_jets, num_classes)`.
    """

    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x
    for idx, filename in enumerate(os.listdir(dirpath)):
        filepath = dirpath + filename
        table = uproot.open(filepath)['tree'].arrays()

        p4 = vector.zip({'px': table['part_px'],
                        'py': table['part_py'],
                        'pz': table['part_pz'],
                        'energy': table['part_energy']})
        
        table['part_pt'] = p4.pt
        table['part_eta'] = p4.eta
        table['part_phi'] = p4.phi
        
        x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
        x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=1)
        y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)
        
        #x_particles = sort_by_feature(x_particles, feature_index=4, descending=True)
        #x_particles = restrict_nb_constituents(x_particles, 32)
        
        name_particles = ''.join(part[5:] for part in particle_features)
        name_jets = ''.join(jet[4:] for jet in jet_features)
        path = f"C://Users//bonvi//Documents//deepset//data//jetclass//{name_particles}_{name_jets}_{max_num_particles}//"
        if not os.path.exists(path):
            os.makedirs(path)
        
        x_particles = np.swapaxes(x_particles, 1, 2)
        print(x_particles.shape, x_jets.shape, y.shape)
        
        np.save(f'{path}//x_px_py_pz_E_pt_eta_phi{max_num_particles}_{idx}.npy', x_particles)
        np.save(f'{path}//x_jets_pt_eta_phi_energy{name_particles}_{max_num_particles}_{idx}.npy', x_jets)
        np.save(f'{path}//y_{idx}.npy', y)

    return x_particles, x_jets, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath',  default = 'D://pc//JetClass//Pythia//train_100M//', help='datasets to download')
    parser.add_argument('--nbr_const', default=200, help='number of constituents')
    parser.add_argument('--features', default=['part_px', 'part_py', 'part_pz', 'part_energy', 'part_pt', 'part_eta', 'part_phi'], help='particules features')
    parser.add_argument('--jet', default=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'], help='jet features')
    args = parser.parse_args()

    read_file(args.filepath, args.nbr_const, args.features, args.jet)