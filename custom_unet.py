"""This python module impements the Unet architecture as defined in https://arxiv.org/pdf/1505.04597.
Only, I use padded convolutions. That way, there is no need for center cropping and the output mask
is the same shape as the input image. 

Additional things: https://towardsdatascience.com/understanding-u-net-61276b10f360
"""

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
    
    def forward(self, x: torch.Tensor):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x_left, x_right):
        return self.conv(torch.cat((x_left, self.upconv(x_right)), dim=1))

class CustomUnet(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, depth: int = 3, start_channels: int = 16) -> None:
        
        super().__init__()
        
        self.encoder_layers = nn.ModuleList([DoubleConv(in_channels, start_channels)])
        for i in range(depth):
            self.encoder_layers.append(DoubleConv(start_channels, start_channels * 2))
            start_channels *= 2
            
        self.decoder_layers = nn.ModuleList()
        for i in range(depth):
            self.decoder_layers.append(Up(start_channels, start_channels // 2))
            start_channels //= 2

        self.output_conv = nn.Conv2d(start_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.encoder_layers[0](x)
        xs = [x]

        for encoding_layer in self.encoder_layers[1:]:
            x = encoding_layer(self.pool(x))
            xs.append(x)
            
        for decoding_layer, x_left in zip(self.decoder_layers, reversed(xs[:-1])):
            x = decoding_layer(x_left, x)

        return self.output_conv(x)

if __name__ == '__main__':
    batch_images = torch.rand(size=(16,3,512,512), device=device)
    model = CustomUnet().to(device)
    out = model(batch_images)
    print(out.shape)