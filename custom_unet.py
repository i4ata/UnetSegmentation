"""This python module impements the Unet architecture as defined in https://arxiv.org/pdf/1505.04597.
Only, I use padded convolutions. That way, there is no need for center cropping and the output mask
is the same shape as the input image. 

Additional things: https://towardsdatascience.com/understanding-u-net-61276b10f360
"""

import torch
import torch.nn as nn
from torchinfo import summary

from model import SegmentationModel
from early_stopper import EarlyStopper

from typing import Tuple, Union, Optional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DiceLoss(nn.Module):
    """Manual implementation of the DICE loss."""
    
    def forward(self, logits: torch.Tensor, mask_true: torch.Tensor):
        logits = torch.sigmoid(logits) > .5
        intersection = (logits * mask_true).sum()
        union = logits.sum() + mask_true.sum()
        return 2 * intersection / union

class DoubleConv(nn.Module):
    """PyTorch module with 2 convolutional layers and ReLU activation"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
    
    def forward(self, x: torch.Tensor):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class Up(nn.Module):
    """PyTorch module that handles the transposed convolution, the residual connections,
    and the subsequent double convolution.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x_left, x_right):
        return self.conv(torch.cat((x_left, self.upconv(x_right)), dim=1))

class UnetModel(nn.Module):
    """Manual implementation of the Unet."""

    def __init__(self, in_channels: int = 3, depth: int = 3, start_channels: int = 16) -> None:
        
        super().__init__()

        self.input_conv = DoubleConv(in_channels, start_channels)

        self.encoder_layers = nn.ModuleList()
        for i in range(depth):
            self.encoder_layers.append(DoubleConv(start_channels, start_channels * 2))
            start_channels *= 2
            
        self.decoder_layers = nn.ModuleList()
        for i in range(depth):
            self.decoder_layers.append(Up(start_channels, start_channels // 2))
            start_channels //= 2

        self.output_conv = nn.Conv2d(start_channels, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.input_conv(x)
        xs = [x]

        for encoding_layer in self.encoder_layers:
            x = encoding_layer(self.pool(x))
            xs.append(x)
            
        for decoding_layer, x_left in zip(self.decoder_layers, reversed(xs[:-1])):
            x = decoding_layer(x_left, x)

        return self.output_conv(x)

class CustomUnet(SegmentationModel):
    """Wrapper of the custom unet to function as a trainable model."""

    def __init__(self,
                 name: str = 'default_name',
                 from_file: bool = True, 
                 image_size: Tuple[int, int] = (320, 320),
                 in_channels: int = 3,
                 start_channels: int = 16,
                 encoder_depth: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        
        super().__init__()

        assert image_size[0] % (2**encoder_depth) == 0
        assert image_size[1] % (2**encoder_depth) == 0

        self.name = name
        self.image_size = image_size
        self.in_channels = in_channels
        self.device = device

        self.save_path = f'models/{name}.pth'

        if from_file:
            self.unet = torch.load(self.save_path, map_location=device)
        else:
            self.unet = UnetModel(in_channels=in_channels, depth=encoder_depth, start_channels=start_channels).to(device)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.loss_fn = lambda logits, masks: self.bce_loss(logits, masks) + self.dice_loss(logits, masks)

    def configure_optimizers(self, **kwargs):
        self.optimizer = torch.optim.Adam(params=self.unet.parameters(), lr=kwargs['lr'])
        self.early_stopper = EarlyStopper(patience=kwargs['patience'])

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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