import albumentations as A
from segmentation_models_pytorch import Unet

# Training data augmentation
train_transform = A.Compose(
    transforms=[
        A.Resize(320, 320),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5)
    ], 
    is_check_shapes=False
)

# Testing data augmentation
val_transform = A.Compose(
    transforms=[
        A.Resize(320, 320) 
    ], 
    is_check_shapes=False
)

def get_pretrained_unet() -> Unet:
    unet = Unet(
        encoder_name='timm-efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=3,
        encoder_depth=5,
        classes=1,
        activation=None
    )
    return unet