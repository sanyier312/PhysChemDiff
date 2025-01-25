import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils.tensor_to_amino_acids import *

class Diffusion(Dataset):
    """
    A PyTorch Dataset class for loading protein sequence feature tensors and extracting biological information.
    """

    def __init__(self, pt_file_path, csv_file_path, test_size=0.2, val_size=0.1):
        """
        Initialize the Diffusion class.

        Parameters:
        pt_file_path (str): Path to the .pt file.
        csv_file_path (str): Path to the CSV file containing biological information.
        test_size (float): Proportion of the dataset to be used for the test set.
        val_size (float): Proportion of the dataset to be used for the validation set.
        """
        self.onehot_tensors, self.feature_tensors, self.masks = self.load_data(pt_file_path)
        self.data = pd.read_csv(csv_file_path)
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_data(test_size, val_size)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.feature_tensors)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at index `idx`.

        Parameters:
        idx (int): Index of the sample.

        Returns:
        tuple: A tuple containing the feature tensor, mask, and label.
        """
        onehot_tensor = self.onehot_tensors[idx]
        feature_tensor = self.feature_tensors[idx]
        mask = self.masks[idx]

        species = str(self.data.iloc[idx]["Species"]) if pd.notna(self.data.iloc[idx]["Species"]) else ""
        keyword = str(self.data.iloc[idx]["Keyword"]) if pd.notna(self.data.iloc[idx]["Keyword"]) else ""
        function = str(self.data.iloc[idx]["Function"]) if pd.notna(self.data.iloc[idx]["Function"]) else ""
        
        # Construct a label by concatenating different biological information
        label = f"Species: {species} Keyword: {keyword} Description: {function}"

        return onehot_tensor, feature_tensor, mask, label

    def load_data(self, file_path):
        """
        Load data from a .pt file.

        Parameters:
        file_path (str): Path to the .pt file.

        Returns:
        tuple: A tuple containing the feature tensors and masks.
        """
        checkpoint = torch.load(file_path)
        onehot_tensors = checkpoint['onehot_tensors']
        feature_tensors = checkpoint['feature_tensors']
        masks = checkpoint['masks']
        return onehot_tensors, feature_tensors, masks

    def split_data(self, test_size, val_size):
        """
        Split the dataset into training, validation, and test sets.

        Parameters:
        test_size (float): Proportion of the dataset to be used for the test set.
        val_size (float): Proportion of the dataset to be used for the validation set.

        Returns:
        tuple: A tuple containing the training, validation, and test subsets.
        """
        indices = np.arange(len(self.feature_tensors))
        train_indices, temp_indices = train_test_split(indices, test_size=test_size + val_size, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size / (test_size + val_size),
                                                     random_state=42)

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)
        test_subset = Subset(self, test_indices)

        return train_subset, val_subset, test_subset

    def get_train_dataloader(self, batch_size=32):
        """
        Get the DataLoader for the training dataset.

        Parameters:
        batch_size (int): The batch size for the DataLoader.

        Returns:
        DataLoader: The training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def get_val_dataloader(self, batch_size=32):
        """
        Get the DataLoader for the validation dataset.

        Parameters:
        batch_size (int): The batch size for the DataLoader.

        Returns:
        DataLoader: The validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    def get_test_dataloader(self, batch_size=32):
        """
        Get the DataLoader for the test dataset.

        Parameters:
        batch_size (int): The batch size for the DataLoader.

        Returns:
        DataLoader: The test DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
