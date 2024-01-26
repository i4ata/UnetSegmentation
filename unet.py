"""This module defines a Unet architecture"""

import torch.nn as nn
import torch
from torchinfo import summary
import segmentation_models_pytorch as smp
from early_stopper import EarlyStopper

from model import SegmentationModel

from typing import Optional, Union, Tuple

class Unet(SegmentationModel):
    """Class that implements a pretrained Unet model. Extends SegmentationModel
    
    Attributes:
        unet: a pretrained neural network with Unet architecture
        loss_fn: the loss function used to train the model
    """
    
    def __init__(self, 
                 name: str = 'default_name',
                 from_file: bool = True,
                 image_size: Tuple[int, int] = (320, 320),
                 encoder_name: str = 'timm-efficientnet-b0',
                 pretrained: bool = True,
                 in_channels: int = 3,
                 encoder_depth: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """Instantiate Unet with `imagenet` pretrained weights.
        Use `Adam` as an optimizer, loss function is the sum of DICE and BCE
        """
        super().__init__()
        
        self.name = name
        self.image_size = image_size
        self.in_channels = in_channels
        self.device = device

        self.save_path = f'models/{name}.pth'

        if from_file:
            self.unet = torch.load(self.save_path, map_location=device)
        else:
            self.unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights='imagenet' if pretrained else None,
                in_channels=in_channels,
                encoder_depth=encoder_depth,
                classes=1,
                activation=None
            ).to(device)

        bce_loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = smp.losses.DiceLoss(mode='binary')
        self.loss_fn = lambda logits, masks: bce_loss_fn(logits, masks) + dice_loss_fn(logits, masks)
    
    def configure_optimizers(self, **kwargs):
        self.optimizer = torch.optim.Adam(params=self.unet.parameters(), lr=kwargs['lr'])
        self.early_stopper = EarlyStopper(patience=kwargs['patience'])

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass"""
        logits = self.unet(images)
        if masks is None:
            return logits
        return logits, self.loss_fn(logits, masks)
    
    def save(self) -> None:
        # Save the whole model, not only the state dict, so that it will work for different unets
        torch.save(self.unet, self.save_path)

    def print_summary(self, batch_size: int = 16) -> None:
        """Summary of model architecture"""
        print(summary(self.unet, input_size=(batch_size, self.in_channels, *self.image_size),
                      col_names=['input_size', 'output_size', 'num_params'],
                      row_settings=['var_names']))
