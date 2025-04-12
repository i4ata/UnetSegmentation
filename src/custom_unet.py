"""
This python module impements the Unet architecture as defined in https://arxiv.org/pdf/1505.04597.
Only, I use padded convolutions. That way, there is no need for center cropping and the output mask
is the same shape as the input image. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(3x3) conv -> ReLU -> (3x3) conv -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
    
    def forward(self, x: torch.Tensor):
        return F.relu(self.conv2(F.relu(self.conv1(x))))
    
class Up(nn.Module):
    """Perform a deconvolution and concatenate with the output of the opposing layer"""

    def __init__(self, in_channels: int, out_channels: int) -> None:

        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((x_left, self.upconv(x_right)), dim=1))

class CustomUnet(nn.Module):

    def __init__(self, in_channels: int, depth: int, start_channels: int) -> None:
        """Structure of the Unet. 
        Each layer in the decoder halves the image size in both dimensions and doubles the number of channels.
        The decoder layer is the opposite.
        """
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.input_conv(x)
        xs = [x]

        for encoding_layer in self.encoder_layers:
            x = encoding_layer(F.max_pool2d(x, 2))
            xs.append(x)
            
        for decoding_layer, x_left in zip(self.decoder_layers, reversed(xs[:-1]), strict=True):
            x = decoding_layer(x_left, x)

        return self.output_conv(x)
