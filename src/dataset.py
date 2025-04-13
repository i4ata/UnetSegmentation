"""This module implements the interaction with the dataset"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import albumentations as A
from PIL import Image
import numpy as np
import os
from typing import Optional, Tuple
import subprocess

from src.utils import train_transform, val_transform

class SegmentationDataset:
    """Fetch the dataset and convert it to usable torch dataloaders"""

    def __init__(self) -> None:
        """Read the dataset either from GitHub or from memory"""

        # Download the data
        data_path = 'Human-Segmentation-Dataset-master'
        if not os.path.exists(data_path) or not os.listdir(data_path):
            print('Downloading dataset')
            subprocess.run(['git', 'clone', 'https://github.com/parth1620/Human-Segmentation-Dataset-master.git'])
        else:
            print('Dataset already downloaded!')
        
        self.df = pd.read_csv(os.path.join('Human-Segmentation-Dataset-master', 'train.csv'))
        
    def get_dataloaders(self, batch_size: int, train_size: float) -> Tuple[DataLoader, DataLoader]:
        """
        Split the data and instantiate dataloaders

        :param batch_size (int): Number of samples per batch
        :param train_size (float): Proportion of the data to keep for training

        :return train_dataloader (torch.utils.data.DataLoader): A directly usable dataloader with the training data
        :return val_dataloader (torch.utils.data.DataLoader): A directly usable dataloader with the validation data
        """

        # Split indices
        train_df = self.df.sample(frac=train_size)
        test_df = self.df.drop(train_df.index)

        # Initializes the train and validation datasets with the corresponding samples and transforms
        train_dataset = _TorchDataset(train_df, train_transform)
        val_dataset = _TorchDataset(test_df, val_transform)

        # Initialize the Pytorch dataloaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

class _TorchDataset(Dataset):
    """An extension of a torch dataset. Used to automatically create dataloaders"""

    def __init__(self, df: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        """
        Initialize the dataset by specifying the samples and the data augmentation

        :param df (pandas.DataFrame): The subset of all samples
        :param augmentation (albumentation.Compose|None): The data augmentation. Specify `None` for no augmentation
        """
        
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Number of available samples"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the `idx`-th item in the dataset. It contains the observation (image) and a label (mask)
        
        :param idx (int): The sample index
        
        :return image: The (optionally augmented) input image as a 3d (CHW) float32 torch tensor
        :return mask: The corresponding mask as a 3d (1HW) long torch tensor with values 1 and 0, where 1 indicates a person
        """

        # Read in the image and the label (mask)
        sample = self.df.iloc[idx]
        image = np.asarray(Image.open(sample['images']))
        mask = np.asarray(Image.open(sample['masks']))

        # Correct grayscale images
        if image.ndim == 2: image = np.tile(image[:,:,np.newaxis], (1,1,3))

        # Perform data augmentation if specified
        if self.augmentation is not None:
            transformed = self.augmentation(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # Normalize image and convert (HWC) -> (CHW)
        image = torch.from_numpy(image).float().permute(2,0,1) / 255.

        # Convert mask to {0, 1}
        mask = torch.from_numpy(mask).float().round().unsqueeze(0) / 255.

        return image, mask
