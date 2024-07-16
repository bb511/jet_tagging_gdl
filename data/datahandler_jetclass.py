import os
import numpy as np
from glob import glob

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
        Process all files matching the file pattern in batches.
        """
        os.chdir(self.directory)
        files = sorted(glob(self.file_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        files_label = sorted(glob("y_*.npy"), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        print(f"Number of files: {len(files)}")
        print(f"Number of label files: {len(files_label)}")

        for i in range(len(files) // 200):
            self.process_batch(i, files[i*200:(i+1)*200], files_label[i*200:(i+1)*200])

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

            data, data_label = self.sort_by_feature(data, data_label)
            data = self.restrict_nb_constituents(data, 32)
            data, data_label = self.cut_transverse_momentum(data, data_label, 0.2)
            data = self.get_features(data, jet_data)

            all_data.append(data)
            all_data_label.append(data_label)

        concatenated_data = np.concatenate(all_data, axis=0)
        concatenated_data_label = np.concatenate(all_data_label, axis=0)
        
        self.plot_non_zero_constituents(concatenated_data, batch_index)
        self.plot_normalised_data(concatenated_data, concatenated_data_label, batch_index)


        name = f"{self.output_name}_{batch_index}"
        np.save(name, concatenated_data)
        np.save(f"y_{name}", concatenated_data_label)

        print(f"Concatenated data saved as {name}.npy")
        print(f"Shape of concatenated data: {concatenated_data.shape}")
    
    def plot_non_zero_constituents(self, data, batch_index):
        """
        Plot the number of non-zero padded constituents in the processed data.

        Args:
            data (np.ndarray): The processed data array.
            batch_index (int): The index of the current batch.
        """
        non_zero_counts = np.sum(np.any(data != 0, axis=1), axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(non_zero_counts, bins=range(0, data.shape[2] + 2, 1), align='left', rwidth=0.8)
        plt.title(f'Number of Non-Zero Constituents (Batch {batch_index})')
        plt.xlabel('Number of Constituents')
        plt.ylabel('Frequency')
        plt.xticks(range(0, data.shape[2] + 1, 2))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        plot_name = f"{self.output_name}_constituents_batch_{batch_index}.png"
        plt.savefig(plot_name)
        plt.close()

        print(f"Constituents plot saved as {plot_name}")
    def plot_normalised_data(self, x_data: np.ndarray, y_data: np.ndarray, batch_index: int):
        """
        Plots the normalised data for each feature and class.

        Args:
            x_data (np.ndarray): The processed data array of shape (batch, num_feat, num_const).
            y_data (np.ndarray): The corresponding labels.
            batch_index (int): The index of the current batch.
        """
        outdir = os.path.join(self.directory, f"normalised_plots_batch_{batch_index}")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        print("Plotting the normalised data...")
        plt.rc("xtick", labelsize=23)
        plt.rc("ytick", labelsize=23)
        plt.rc("axes", titlesize=25)
        plt.rc("axes", labelsize=25)
        plt.rc("legend", fontsize=22)
        
        x_data_seg = self.segregate_data(x_data, y_data)
        colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
        data_classes = ["Gluon", "Quark", "W", "Z", "Top"]
        feature_labels = self.select_feature_labels()
        
        print(feature_labels)
        for feature in range(x_data.shape[1]):  # Iterate over features
            plt.figure(figsize=(12, 10))
            for data_class_idx in range(len(x_data_seg)):
                plt.hist(
                    x=x_data_seg[data_class_idx][:, feature, :].flatten(),  # Flatten across batch and constituents
                    bins=60,
                    alpha=0.5,
                    histtype="step",
                    linewidth=2.5,
                    label=data_classes[data_class_idx],
                    density=True,
                    color=colors[data_class_idx],
                )
            plt.xlabel(feature_labels[feature])
            plt.ylabel("Probability Density")
            plt.gca().set_yscale("log")
            plt.legend()
            plt.savefig(
                os.path.join(outdir, f"{feature_labels[feature]}_batch_{batch_index}.pdf")
            )
            plt.close()
        
        print(f"Plots saved to: {outdir}")

    def segregate_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Segregates the data based on class labels.

        Args:
            x_data (np.ndarray): The processed data array of shape (batch, num_feat, num_const).
            y_data (np.ndarray): The corresponding labels.

        Returns:
            list: List of arrays, each containing data for a specific class.
        """
        unique_classes = np.unique(y_data)
        return [x_data[y_data == cls] for cls in unique_classes]

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
        sort_indices = np.argsort(data[:, feature_index, :], axis=1)
        if descending:
            sort_indices = np.flip(sort_indices, axis=1)
        return data[np.arange(data.shape[0])[:, None], :, sort_indices], labels[np.arange(labels.shape[0])[:, None], sort_indices]

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
        return np.pad(x_data[:, :, :nconst], ((0, 0), (0, 0), (0, max(0, nconst - x_data.shape[2]))))

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
        mask = x_data[:, 4, :] > min_pt
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
        jet_data = np.expand_dims(jet_data, 2)
        data[:, 6:, :] -= jet_data[:, 1:3, :]
        return data[:, [5, 6, 7], :]

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
        return data[:, [0, 1, 2], :]

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
        return data[:, [0, 1, 2, 3], :]

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
        jet_data = np.stack((jet_px, jet_py, jet_pz), axis=1)[:, :, np.newaxis]
        return np.concatenate((data[:, [0, 1, 2], :], jet_data), axis=1)

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
        jet_data = np.stack((jet_px, jet_py, jet_pz, jet_E), axis=1)[:, :, np.newaxis]
        return np.concatenate((data[:, [0, 1, 2, 3], :], jet_data), axis=1)

if __name__ == "__main__":
    directory = "C://Users//bonvi//Documents//deepset//data//jetclass//ptetaphi_ptetaphi_32//"
    processor = JetDataProcessor(directory, "x_pt_etarel_phirel_32_*.npy", "concatenated_x_pt_etarel_phirel_32", "ptetaphi")
    processor.process_files()