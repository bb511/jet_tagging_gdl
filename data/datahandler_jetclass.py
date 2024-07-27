import os
import numpy as np
from glob import glob
from sklearn.preprocessing import RobustScaler

from matplotlib import pyplot as plt

class JetDataProcessor:
    """
    A class for processing jet data files, applying various transformations and feature selections.
    """

    def __init__(self, directory, file_pattern, output_name, feat):
        """
        Initialize the JetDataProcessor.

        Args:
            directory (str): The directory containing the data files.
            file_pattern (str): The pattern to match data files.
            output_name (str): The base name for output files.
            feat (str): The feature selection method to use.
        """
        self.directory = directory
        self.file_pattern = file_pattern
        self.output_name = output_name
        self.feat = feat
        self.validate_feat()

    def validate_feat(self):
        """
        Validate the selected feature method.

        Raises:
            ValueError: If the selected feature method is not valid.
        """
    
        valid_feats = ["ptetaphi", "pxpypz", "pxpypzE", "pxpypzjet", "pxpypzEjet"]
        if self.feat not in valid_feats:
            raise ValueError(f"Feature selection not valid. Choose from {valid_feats}")

    def process_files(self):
        """
        Process all files matching the file pattern in batches, ensuring equal label distribution.
        """
        os.chdir(self.directory)
        files = sorted(glob(self.file_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        files_label = sorted(glob("y_*.npy"), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        print(f"Number of files: {len(files)}")
        print(f"Number of label files: {len(files_label)}")

        # Assuming there are 10 different labels (0-9)
        num_labels = 10
        files_per_label = len(files) // num_labels
        batch_size = 200

        for i in range(len(files) // batch_size):
            batch_files = []
            batch_label_files = []
            
            for label in range(num_labels):
                start_idx = label * files_per_label + (i * batch_size // num_labels)
                end_idx = start_idx + (batch_size // num_labels)
                
                batch_files.extend(files[start_idx:end_idx])
                batch_label_files.extend(files_label[start_idx:end_idx])
            
            self.process_batch(i, batch_files, batch_label_files)

    def process_batch(self, batch_index, batch_files, batch_label_files):
        """
        Process a batch of files, applying transformations and saving the results.

        Args:
            batch_index (int): The index of the current batch.
            batch_files (list): List of data files to process.
            batch_label_files (list): List of corresponding label files.
        """
        print(f"Processing batch {batch_index}")
        all_data = []
        all_data_label = []

        for j, (file, label_file) in enumerate(zip(batch_files, batch_label_files)):
            data = np.load(file)
            data_label = np.load(label_file)
            jet_data = np.load(f"x_jets_pt_eta_phi_energypxpypzenergyptetaphi_200_{batch_index*200+j}.npy")

            data = self.get_features(data, jet_data)

            all_data.append(data)
            all_data_label.append(data_label)

        concatenated_data = np.concatenate(all_data, axis=0)
        concatenated_data_label = np.concatenate(all_data_label, axis=0)
        
        self.plot_non_zero_constituents(concatenated_data, batch_index)

        # Apply robust normalization
        scaler = RobustScaler()
        original_shape = concatenated_data.shape
        flattened_data = concatenated_data.reshape(-1, original_shape[-1])
        normalized_data = scaler.fit_transform(flattened_data)
        normalized_data = normalized_data.reshape(original_shape)
        

        feat_dir = os.path.join(self.directory, f'{self.feat}_robust_normalized')
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        self.plot_normalised_data(normalized_data, concatenated_data_label, batch_index)


        # Save normalized data in the 'feat_robust_normalized' directory
        name = f"{self.output_name}_robust_normalized_{batch_index}"
        np.save(os.path.join(feat_dir, name), normalized_data)
        np.save(os.path.join(feat_dir, f"y_{name}"), concatenated_data_label)

        print(f"Normalized data saved as {os.path.join(feat_dir, name)}.npy")
        print(f"Shape of normalized data: {normalized_data.shape}")
    
    def count_constituents_per_jet(self, x_data: np.array):
        """Count the number of non-zero constituents for each of the jets in the dataset."""
        constituents_distribution = []
        for jet in x_data:
            const_nb = 0
            for const in jet:
                if const[0] != 0:
                    const_nb += 1
            constituents_distribution.append(const_nb)
        return constituents_distribution

    def plot_non_zero_constituents(self, data, batch_index):
        """
        Plot the number of non-zero padded constituents in the processed data.

        Args:
            data (np.ndarray): The processed data array.
            batch_index (int): The index of the current batch.
        """
        outdir = os.path.join(self.directory, f"constituents_plots_batch_{batch_index}")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        print("Plotting the number of constituents in data...")
        plt.rc("xtick", labelsize=23)
        plt.rc("ytick", labelsize=23)
        plt.rc("axes", titlesize=25)
        plt.rc("axes", labelsize=25)
        plt.rc("legend", fontsize=22)

        constituents_distribution = self.count_constituents_per_jet(data)
        median_const = np.median(constituents_distribution)
        average_const = np.average(constituents_distribution)
        max_const = np.max(constituents_distribution)

        plt.figure(figsize=(12, 10))
        plt.hist(
            x=constituents_distribution,
            bins=150,
            range=(0, max_const),
            alpha=0.4,
            histtype="step",
            linewidth=2.5,
            label=f"Median: {median_const:.3f}\nAverage: {average_const:.3f}",
            color="#648FFF",
        )
        plt.xlabel("Number of Constituents")
        plt.ylabel("Number of Jets")
        plt.legend()
        plt.gca().set_yscale("log")
        plt.xlim(0, max_const)

        plot_name = f"{self.output_name}_constituents_batch_{batch_index}.pdf"
        plt.savefig(os.path.join(outdir, plot_name))
        plt.close()

        print(f"Constituents plot saved as {os.path.join(outdir, plot_name)}")

    def plot_normalised_data(self, x_data: np.ndarray, y_data: np.ndarray, batch_index: int):
        """
        Plots the normalised data for each feature and class.
        Args:
            x_data (np.ndarray): The processed data array of shape (batch, num_const, num_feat).
            y_data (np.ndarray): The corresponding labels.
            batch_index (int): The index of the current batch.
        """
        outdir = os.path.join(self.directory, f"normalised_plots_batch_{batch_index}")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    
        print("Plotting the normalised data...")
        plt.rc("xtick", labelsize=12)
        plt.rc("ytick", labelsize=12)
        plt.rc("axes", titlesize=14)
        plt.rc("axes", labelsize=14)
        plt.rc("legend", fontsize=10)
    
        x_data_seg = self.segregate_data(x_data, y_data)
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        data_classes = [
            "QCD", "H → b¯b", "H → c¯c", "H → gg", "H → 4q",
            "H → ℓνqq′", "t → bqq′", "t → bℓν", "W → qq′", "Z → q¯q"
        ]
        feature_labels = self.select_feature_labels()
    
        print(feature_labels)
        print(f"Number of features: {x_data.shape[2]}")
        for feature in range(x_data.shape[2]):  # Iterate over features
            plt.figure(figsize=(16, 10))
            for data_class_idx in range(len(x_data_seg)):
                if data_class_idx < len(colors) and data_class_idx < len(data_classes):
                    plt.hist(
                        x=x_data_seg[data_class_idx][:, :, feature].flatten(),  # Flatten across batch and constituents
                        bins=60,
                        alpha=0.5,
                        histtype="step",
                        linewidth=2.5,
                        label=data_classes[data_class_idx],
                        density=True,
                        color=colors[data_class_idx],
                    )
            plt.xlabel(feature_labels[feature] if feature < len(feature_labels) else f"Feature {feature}")
            plt.ylabel("Probability Density")
            plt.title(f"Distribution of {feature_labels[feature] if feature < len(feature_labels) else f'Feature {feature}'}")
            plt.gca().set_yscale("log")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(
                os.path.join(outdir, f"{feature_labels[feature] if feature < len(feature_labels) else f'Feature_{feature}'}_batch_{batch_index}.pdf"),
                bbox_inches='tight'
            )
            plt.close()
    
        print(f"Plots saved to: {outdir}")

    def segregate_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Segregates the data based on class labels.
        Args:
            x_data (np.ndarray): The processed data array of shape (batch, num_const, num_feat).
            y_data (np.ndarray): The corresponding labels.
        Returns:
            list: List of arrays, each containing data for a specific class.
        """
        
        num_classes = y_data.shape[1]
        
        segregated_data = []
        for cls in range(num_classes):
            mask = y_data[:, cls] == 1
            class_data = x_data[mask]
            segregated_data.append(class_data)
            print(f"Class {cls}: {class_data.shape}")
            
        return segregated_data

    def select_feature_labels(self):
        """
        Selects the appropriate feature labels based on the feature selection method.

        Returns:
            list: List of feature labels.
        """
        feature_labels = {
            "ptetaphi": ["pt", "eta_rel", "phi_rel"],
            "pxpypz": ["px", "py", "pz"],
            "pxpypzjet": ["px", "py", "pz"],
            "pxpypzE": ["px", "py", "pz", "E"],
            "pxpypzEjet": ["px", "py", "pz", "E"],
        }
        return feature_labels.get(self.feat)

    @staticmethod
    def sort_by_feature(data, labels, feature_index=4, descending=True):
        """
        Sort the data and labels based on a specific feature.

        Args:
            data (np.ndarray): The input data array.
            labels (np.ndarray): The corresponding labels.
            feature_index (int): The index of the feature to sort by.
            descending (bool): Whether to sort in descending order.

        Returns:
            tuple: Sorted data and labels.
        """
        sort_indices = np.argsort(data[:, :, feature_index], axis=1)
        if descending:
            sort_indices = np.flip(sort_indices, axis=1)
        return data[np.arange(data.shape[0])[:, None], sort_indices, :], labels[np.arange(labels.shape[0])[:, None], sort_indices]

    @staticmethod
    def restrict_nb_constituents(x_data, nconst):
        """
        Restrict the number of constituents in each jet to a fixed number.

        Args:
            x_data (np.ndarray): The input data array.
            nconst (int): The number of constituents to keep.

        Returns:
            np.ndarray: Data array with restricted number of constituents.
        """
        return np.pad(x_data[:, :nconst, :], ((0, 0), (0, max(0, nconst - x_data.shape[1])), (0, 0)))

    @staticmethod
    def cut_transverse_momentum(x_data, y_data, min_pt):
        """
        Apply a transverse momentum cut to the data.

        Args:
            x_data (np.ndarray): The input data array.
            y_data (np.ndarray): The corresponding labels.
            min_pt (float): The minimum transverse momentum threshold.

        Returns:
            tuple: Filtered data and labels.
        """
        mask = x_data[:, :, 4] > min_pt
        structure_memory = mask.sum(axis=1)
        x_data = [jet[mask[i]] for i, jet in enumerate(x_data) if mask[i].any()]
        y_data = y_data[structure_memory > 0]
        return x_data, y_data

    def get_features(self, data, jet_data):
        """
        Apply the selected feature extraction method.

        Args:
            data (np.ndarray): The input data array.
            jet_data (np.ndarray): The jet data array.

        Returns:
            np.ndarray: Extracted features.
        """
        feature_functions = {
            "ptetaphi": self.select_features_ptetaphi,
            "pxpypz": self.select_features_pxpypz,
            "pxpypzE": self.select_features_pxpypzE,
            "pxpypzjet": self.select_features_pxpypzjet,
            "pxpypzEjet": self.select_features_pxpypzEjet
        }
        return feature_functions[self.feat](data, jet_data)

    @staticmethod
    def select_features_ptetaphi(data, jet_data):
        """
        Select pt, eta, phi features.

        Args:
            data (np.ndarray): The input data array.
            jet_data (np.ndarray): The jet data array.

        Returns:
            np.ndarray: Selected features.
        """
        # Ensure jet_data has the right shape for broadcasting
        jet_data = jet_data[:, np.newaxis, :]
        
        # Subtract eta and phi of jet from constituents
        data[:, :, 5] -= jet_data[:, :, 1]  # eta
        data[:, :, 6] -= jet_data[:, :, 2]  # phi
        
        return data[:, :, [4, 5, 6]]

    @staticmethod
    def select_features_pxpypz(data, jet_data):
        """
        Select px, py, pz features.

        Args:
            data (np.ndarray): The input data array.
            jet_data (np.ndarray): The jet data array.

        Returns:
            np.ndarray: Selected features.
        """
        return data[:, :, [0, 1, 2]]

    @staticmethod
    def select_features_pxpypzE(data, jet_data):
        """
        Select px, py, pz, E features.

        Args:
            data (np.ndarray): The input data array.
            jet_data (np.ndarray): The jet data array.

        Returns:
            np.ndarray: Selected features.
        """
        return data[:, :, [0, 1, 2, 3]]

    @staticmethod
    def select_features_pxpypzjet(data, jet_data):
        """
        Select px, py, pz features and add jet features.

        Args:
            data (np.ndarray): The input data array.
            jet_data (np.ndarray): The jet data array.

        Returns:
            np.ndarray: Selected features with jet data.
        """
        jet_px = jet_data[:, 0] * np.cos(jet_data[:, 2])
        jet_py = jet_data[:, 0] * np.sin(jet_data[:, 2])
        jet_pz = jet_data[:, 0] * np.sinh(jet_data[:, 1])
        jet_data = np.stack((jet_px, jet_py, jet_pz), axis=1)[:, np.newaxis, :]
        return np.concatenate((data[:, :, [0, 1, 2]], jet_data), axis=1)

    @staticmethod
    def select_features_pxpypzEjet(data, jet_data):
        """
        Select px, py, pz, E features and add jet features.

        Args:
            data (np.ndarray): The input data array.
            jet_data (np.ndarray): The jet data array.

        Returns:
            np.ndarray: Selected features with jet data.
        """
        jet_px = jet_data[:, 0] * np.cos(jet_data[:, 2])
        jet_py = jet_data[:, 0] * np.sin(jet_data[:, 2])
        jet_pz = jet_data[:, 0] * np.sinh(jet_data[:, 1])
        jet_E = jet_data[:, 3]
        jet_data = np.stack((jet_px, jet_py, jet_pz, jet_E), axis=1)[:, np.newaxis, :]
        return np.concatenate((data[:, :, [0, 1, 2, 3]], jet_data), axis=1)


if __name__ == "__main__":
    directory = "C://Users//bonvi//Documents//deepset//data//jetclass//pxpypzenergyptetaphi_ptetaphienergy_200//"
    processor = JetDataProcessor(directory, "x_px_py_pz_E_pt_eta_phi200_*.npy", "concatenated_x_pt_etarel_phirel_32", "pxpypzE")
    processor.process_files()