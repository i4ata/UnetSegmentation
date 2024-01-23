"""This module implements the interaction with the dataset"""

import os
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2 as cv

from typing import Optional, Tuple

class SegmentationDataset:
    """Class that handles the functionality of semantic segmentation dataset

    Attributes:
        df: pandas data frame that stores the image file locations and their respective mask file locations
        train_dataset, test_dataset: PyTorch datasets, initialized after the dataset is split into train and test by invoking split()
        train_dataloader, test_dataloader: PyTorch dataloaders, initialized after the dataset is split and get_dataloaders() is invoked

    Methods:
        split: split `df` and instatiate the training and testing datasets with their respective transforms
        get_dataloaders: instantiate the training and testing dataloaders from the training and testing datasets
    """

    def __init__(self) -> None:
        """
        Create a segmentation dataset.
        If the dataset is not in memory, download it from GitHub.
        Load the data.
        """

        if not os.path.exists('Human-Segmentation-Dataset-master/'):
            print('Downloading dataset')
            os.system('git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git')
        else:
            print('Dataset already downloaded!')
            
        self.df = pd.read_csv('Human-Segmentation-Dataset-master/train.csv')
        self.dataset = TorchDataset(self.df)
    
    def split(self, 
              train_size: float = .8, 
              train_transform: Optional[A.Compose] = None, 
              test_transform: Optional[A.Compose] = None, 
              random_state: Optional[int] = 0) -> None:
        """Split the data into training and testing dataset

        Args:
            train_size: the proportion of data used for training: (0, 1)
            random_state: to ensure reproducibility
        """

        train_df = self.df.sample(frac=train_size, random_state=random_state)
        test_df = self.df.drop(train_df.index)
        self.train_dataset = TorchDataset(train_df, train_transform)
        self.test_dataset = TorchDataset(test_df, test_transform)

    def get_dataloaders(self, batch_size: int = 32) -> None:
        """After the data is split, instantiate the data loaders
        
        Args:
            batch_size: size of 1 batch
        """

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size)

class TorchDataset(Dataset):
    """This class is an extension of a standard PyTorch dataset that includes data augmentation
    
    Attributes:
        df: The pandas dataframe containing the images and masks file locations
        augmentation: an Albumentations augmentation procedure

    Methods:
        __len__: get the number of samples in the dataset
        __getitem__: get the transformed image and mask at that index
    """

    def __init__(self, df: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        """Create a torch dataset
        
        Args:
            df: pandas data frame that stores the file paths of the images and the masks
            augmentation: data augmentation
        """
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Get the number of samples in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and mask at the index"""
        sample = self.df.iloc[idx]
        
        image = cv.imread(sample['images'])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        mask = cv.imread(sample['masks'], cv.IMREAD_GRAYSCALE)

        if self.augmentation is not None:
            transformed = self.augmentation(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # Normalize image and convert (HWC) -> (CHW)
        image = torch.from_numpy(image).float() / 255.
        image = image.permute(2,0,1)

        # Convert mask to {0, 1}
        mask = torch.from_numpy(mask).float() / 255.
        mask = torch.round(mask)
        mask = mask.unsqueeze(0)

        return image, mask
