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
    
class CustomUnet(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        
        super().__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up1 = DoubleConv(1024, 512)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up2 = DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up3 = DoubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up4 = DoubleConv(128, 64)

        self.outpu_conv = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.down5(self.pool(x4))

        x = self.upconv1(x5)
        x = self.up1(torch.cat((x4, x), dim=1))
        x = self.upconv2(x)
        x = self.up2(torch.cat((x3, x), dim=1))
        x = self.upconv3(x)
        x = self.up3(torch.cat((x2, x), dim=1))
        x = self.upconv4(x)
        x = self.up4(torch.cat((x1, x), dim=1))

        return self.outpu_conv(x)

if __name__ == '__main__':
    batch_images = torch.rand(size=(16,3,512,512), device=device)
    model = CustomUnet().to(device)
    out = model(batch_images)
    print(out.shape)