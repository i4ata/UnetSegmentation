import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2 as cv

from typing import Optional, Tuple, Union, Literal, Dict

# (H x W) of the input images
IMAGE_SIZE = (320, 320)

# Training data augmentation
TRAIN_TRANSFORM = A.Compose(
    transforms=[
        A.Resize(*IMAGE_SIZE),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5)
    ], 
    is_check_shapes=False
)

# Testing data augmentation
TEST_TRANSFORM = A.Compose(
    transforms=[
        A.Resize(*IMAGE_SIZE), 
    ], 
    is_check_shapes=False
)

def get_image(
        img_path: Union[str, Path], 
        transform: Optional[Union[A.Compose, Literal['train', 'test']]] = None, 
        mask_path: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Union[None, torch.Tensor, np.ndarray]]]:
    """Read an image from memory, apply transformations, and conver to PyTorch tensor
    
    Args:
        img_path: the path to the image
        transform: the data augmentation procedure. Can be omitted
        mask_path: the path to the mask file. Can be omitted

    Returns:
        dict(
            original=dict(
                image: <the original image as a NumPy array>
                mask: <the original mask as a NumPy array, if passed>    
            ),
            transformed=dict(
                image: <the image as a PyTorch tensor, transformed according to `transform`>
                image: <the mask as a PyTorch tensor, if passed, transformed according to `transform`>
                
            )
        )
    """
    if isinstance(transform, str):
        if transform == 'train':
            transform = TRAIN_TRANSFORM
        elif transform == 'test':
            transform = TEST_TRANSFORM
        else:
            transform = None 

    original_image = cv.imread(img_path)
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

    original_mask = None if mask_path is None else cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

    image_transformed = original_image.copy()
    mask_transformed = None if original_mask is None else original_mask.copy()

    if transform is not None:
        if original_mask is not None:
            transformed = transform(image=original_image, mask=original_mask)
            image_transformed, mask_transformed = transformed['image'], transformed['mask']
        else:
            image_transformed = transform(image=original_image)['image']

    # Normalize image and convert (HWC) -> (CHW)
    image_transformed = torch.Tensor(image_transformed) / 255.
    image_transformed = image_transformed.permute(2,0,1)

    if mask_transformed is not None:
        # Convert mask to {0, 1}
        mask_transformed = torch.Tensor(mask_transformed) / 255
        mask_transformed = torch.round(mask_transformed)
        mask_transformed = mask_transformed.unsqueeze(0)

    return {
        'original': {
            'image': original_image,
            'mask': original_mask
        },
        'transformed': {
            'image': image_transformed,
            'mask': mask_transformed
        }
    }

class SegmentationDataset:
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
            print('Dataset aready downloaded!')
            
        self.df = pd.read_csv('Human-Segmentation-Dataset-master/train.csv')
    
    def split(self, train_size: float = .8, random_state: Optional[int] = 0) -> None:
        """Split the data into training and testing dataset

        Args:
            train_size: the proportion of data used for training: (0, 1)
            random_state: to ensure reproducibility
        """


        train_df = self.df.sample(frac=train_size, random_state=random_state)
        test_df = self.df.drop(train_df.index)
        self.train_dataset = TorchDataset(train_df, TRAIN_TRANSFORM)
        self.test_dataset = TorchDataset(test_df, TEST_TRANSFORM)

    def get_dataloaders(self, batch_size: int = 32) -> None:
        """After the data is split, get data loaders
        
        Args:
            batch_size: size of 1 batch
        """

        self.train_dataloader = DataLoader(dataset=self.train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
        
        self.test_dataloader = DataLoader(dataset=self.test_dataset, 
                                          batch_size=batch_size)


class TorchDataset(Dataset):

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
        transformed = get_image(img_path=sample['images'], transform=self.augmentation, mask_path=sample['masks'])['transformed']
        return transformed['image'], transformed['mask']