"""
This helper file defines some utilities to use in any script
Namely, the augmentations as well as fetching a pretrained U-Net from segmentation_models_pytorch
"""

import albumentations as A
from segmentation_models_pytorch import Unet
import yaml

# Get the parameters regarding the augmentation
with open('params.yaml') as f:
    params = yaml.safe_load(f)['augmentation']

# Training data augmentation
train_transform = A.Compose(
    transforms=[
        A.Resize(*params['resize']),
        A.HorizontalFlip(p=params['horizontal_flip']),
        A.VerticalFlip(p=params['horizontal_flip'])
    ], 
    is_check_shapes=False
)

# Validation data augmentation
val_transform = A.Compose(
    transforms=[
        A.Resize(*params['resize'])
    ],
    is_check_shapes=False
)

def get_pretrained_unet() -> Unet:
    """
    Retrieve a pretrained U-Net from `segmentation_models_pytorch`

    :return unet (Unet): The pretrained model that can be directly used as a `torch.nn.Module`
    """
    return Unet(encoder_name='timm-efficientnet-b0', in_channels=3, encoder_depth=5, classes=1)
