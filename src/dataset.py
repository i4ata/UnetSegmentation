"""This module implements the interaction with the dataset"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import albumentations as A
from PIL import Image
import numpy as np
import os
from typing import Optional, Tuple

from src.utils import train_transform, val_transform

class SegmentationDataset:

    def __init__(self) -> None:

        if not os.path.exists('Human-Segmentation-Dataset-master/'):
            print('Downloading dataset')
            os.system('git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git')
        else:
            print('Dataset already downloaded!')
            
        self.df = pd.read_csv('Human-Segmentation-Dataset-master/train.csv')
        self.dataset = TorchDataset(self.df)
    
    def split(self, train_size: float = .8, random_state: Optional[int] = 0) -> None:

        train_df = self.df.sample(frac=train_size, random_state=random_state)
        test_df = self.df.drop(train_df.index)

        self.train_dataset = TorchDataset(train_df, train_transform)
        self.val_dataset = TorchDataset(test_df, val_transform)

    def get_dataloaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:

        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

class TorchDataset(Dataset):

    def __init__(self, df: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:
        
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sample = self.df.iloc[idx]
        
        image = np.asarray(Image.open(sample['images']))
        mask = np.asarray(Image.open(sample['masks']))

        if image.ndim == 2: image = np.tile(image[:,:,np.newaxis], (1,1,3))

        if self.augmentation is not None:
            transformed = self.augmentation(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # Normalize image and convert (HWC) -> (CHW)
        image = torch.from_numpy(image).float().permute(2,0,1) / 255.

        # Convert mask to {0, 1}
        mask = torch.from_numpy(mask).float().round().unsqueeze(0) / 255.

        return image, mask

from tqdm import tqdm
from glob import glob
if __name__ == '__main__':
    d = SegmentationDataset().dataset
    print(d.df.iloc[242].values)
    # print(d[242])
    # for i in tqdm(range(len(d))):
    #     try:
    #         image, mask = d[i]
    #     except RuntimeError as e:
    #         print(d.df.iloc[i])
    #         print(np.asarray(Image.open(d.df.iloc[i]['images'])).shape)
        # print(image.shape, mask.shape)
    # for image_path in glob('Human-Segmentation-Dataset-master/Training_Images/*.jpg'): print(image_path, np.asarray(Image.open(image_path)).ndim)