"""This module implements the interaction with the dataset"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import albumentations as A
import cv2 as cv

import os
from typing import Optional, Tuple

class SegmentationDataset:

    def __init__(self) -> None:

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

        train_df = self.df.sample(frac=train_size, random_state=random_state)
        test_df = self.df.drop(train_df.index)
        self.train_dataset = TorchDataset(train_df, train_transform)
        self.test_dataset = TorchDataset(test_df, test_transform)

    def get_dataloaders(self, batch_size: int = 32) -> None:

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size)

class TorchDataset(Dataset):

    def __init__(self, df: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:
        
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
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
