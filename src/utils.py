import albumentations as A
from segmentation_models_pytorch import Unet
import yaml

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

# Testing data augmentation
val_transform = A.Compose(
    transforms=[
        A.Resize(*params['resize'])
    ],
    is_check_shapes=False
)

# Function to retrieve the pretrained Unet model
def get_pretrained_unet() -> Unet:
    return Unet(encoder_name='timm-efficientnet-b0', in_channels=3, encoder_depth=5, classes=1)
