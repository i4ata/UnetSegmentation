import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2 as cv
from typing import Optional, Tuple, Union

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = (320, 320)

train_transform = A.Compose(transforms=[
    A.Resize(*image_size),
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5)
], is_check_shapes=False)

test_transform = A.Compose(transforms=[
    A.Resize(*image_size), 
], is_check_shapes=False)

def get_image(img_path: str, 
              transform: Optional[Union[A.Compose, str]] = None, 
              mask_path: Optional[str] = None) -> dict:
    
    if isinstance(transform, str):
        if transform == 'train':
            transform = train_transform
        elif transform == 'test':
            transform = test_transform
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

    image_transformed = torch.Tensor(image_transformed) / 255.
    image_transformed = image_transformed.permute(2,0,1)

    if mask_transformed is not None:    
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

class SegmentationDataset():
    def __init__(self) -> None:
        if not os.path.exists('Human-Segmentation-Dataset-master/'):
            print('Downloading dataset')
            os.system('git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git')
        else:
            print('Dataset aready downloaded!')
            
        self.df = pd.read_csv('Human-Segmentation-Dataset-master/train.csv')
    
    def split(self, train_size: float = .8, random_state: Optional[int] = 0) -> None:
        
        train_df = self.df.sample(frac=train_size, random_state=random_state)
        test_df = self.df.drop(train_df.index)
        self.train_dataset = TorchDataset(train_df, train_transform)
        self.test_dataset = TorchDataset(test_df, test_transform)

    def get_dataloaders(self, batch_size: int = 32) -> None:

        self.train_dataloader = DataLoader(dataset=self.train_dataset, 
                                           batch_size=batch_size, 
                                           pin_memory=device=='cuda', 
                                           shuffle=True)
        
        self.test_dataloader = DataLoader(dataset=self.test_dataset, 
                                          batch_size=batch_size,
                                          pin_memory=device=='cuda')


class TorchDataset(Dataset):

    def __init__(self, df: pd.DataFrame, augmentation: Optional[A.Compose]) -> None:
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        sample = self.df.iloc[idx]
        transformed = get_image(img_path=sample['images'], transform=self.augmentation, mask_path=sample['masks'])['transformed']
        return transformed['image'], transformed['mask']