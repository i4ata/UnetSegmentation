"""
This python module impements the Unet architecture as defined in https://arxiv.org/pdf/1505.04597.
Only, I use padded convolutions. That way, there is no need for center cropping and the output mask is the same shape as the input image. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(3x3) conv -> ReLU -> (3x3) conv -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Randomly initialize the layers for this module (blue arrow in diagram)
        
        :param in_channels: The output channels of the previous feature map
        :param out_channels: The size of the produced feature map
        """
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
    
    def forward(self, x: torch.Tensor):
        """
        Simply pass the input through the layers

        :param x (torch.Tensor): a 4d (BCHW) float32 batched feature map, where C = `in_channels`
        
        :return y (torch.Tensor): a 4d (BC'HW) float32 batched feature map, where C' = `out_channels` 
        """

        return F.relu(self.conv2(F.relu(self.conv1(x))))
    
class Up(nn.Module):
    """Perform a deconvolution and concatenate with the output of the opposing layer"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Tandomly initialize the layers of this module

        :param in_channels: The output channels of the previous feature map
        :param out_channels: The size of the produced feature map
        """

        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2) # Green arrow
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels) # Blue arrow

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the layers and concatenate with the opposing feature map

        :param x_left (torch.Tensor): 4d (BCHW) f32, the produced feature map in the corresponding encoder layer
        :param x_right (torch.Tensor): 4d (BCHW) f32, the input at the current stage
        """

        return self.conv(
            torch.cat((x_left, self.upconv(x_right)), dim=1) # Grey arrow
        )

class CustomUnet(nn.Module):
    """The entire U-Net architecture"""

    def __init__(self, in_channels: int, depth: int, start_channels: int) -> None:
        """
        Structure of the Unet. 
        Each layer in the decoder halves height and width of the image and doubles the number of channels.
        The decoder layer is the opposite.

        :param in_channels (int): The number of image channels. Normally it's 3 (RGB) but it would work with any
        :param depth (int): How many times to encode and decode the image
        :param start_channels (int): With how many channels to start. Each subsequent encoder layer doubles the channels 
        """
        
        super().__init__()

        # First layer map to `start_channels`
        self.input_conv = DoubleConv(in_channels, start_channels)

        # Encoder. Double the channels at each layer
        self.encoder_layers = nn.ModuleList()
        for i in range(depth):
            self.encoder_layers.append(DoubleConv(start_channels, start_channels * 2))
            start_channels *= 2
            
        # Decoder. Halve the channels at each layer
        self.decoder_layers = nn.ModuleList()
        for i in range(depth):
            self.decoder_layers.append(Up(start_channels, start_channels // 2))
            start_channels //= 2

        # Final output is a 1:1 convolution. Map to a single channel since we have 1 class
        self.output_conv = nn.Conv2d(start_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass an image through the U-Net

        :param x: 4d (BCHW) f32 batch of images
        
        :return y: 4d (B1HW) f32 batch of probability distributions over the number of classes (1) for each pixel in each image
        """

        x = self.input_conv(x)
        
        # To keep track of the encodings
        xs = [x]

        # Encode the image
        for encoding_layer in self.encoder_layers:
            x = encoding_layer(F.max_pool2d(x, 2)) # Max pool -> red arrow
            xs.append(x) # Store the feature map
            
        # Decode the image while concatenating with the opposing encoder layer at each step
        for decoding_layer, x_left in zip(self.decoder_layers, reversed(xs[:-1])):
            x = decoding_layer(x_left, x)

        # 1:1 convolution into probability distributions over the classes for each pixel
        return self.output_conv(x)
