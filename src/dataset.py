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

    def __init__(self) -> None:

        # Download the data
        data_path = 'Human-Segmentation-Dataset-master'
        if not os.path.exists(data_path) or not os.listdir(data_path):
            print('Downloading dataset')
            subprocess.run(['git', 'clone', 'https://github.com/parth1620/Human-Segmentation-Dataset-master.git'])
        else:
            print('Dataset already downloaded!')
        self.df = pd.read_csv(os.path.join('Human-Segmentation-Dataset-master', 'train.csv'))
        
    def get_dataloaders(self, batch_size: int, train_size: float) -> Tuple[DataLoader, DataLoader]:

        # Split indices
        train_df = self.df.sample(frac=train_size)
        test_df = self.df.drop(train_df.index)

        # Initializes the train and validation datasets with the corresponding samples and transforms
        train_dataset = TorchDataset(train_df, train_transform)
        val_dataset = TorchDataset(test_df, val_transform)

        # Initialize the Pytorch dataloaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

class TorchDataset(Dataset):

    def __init__(self, df: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:        
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
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
