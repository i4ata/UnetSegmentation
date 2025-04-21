import torch
import torch.nn as nn
from torch.nn.functional import pad

class Conv2dManual(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super(Conv2dManual, self).__init__()
        self.k = kernel_size
        self.s = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.k, self.k))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        h_, w_ = h // self.s, w // self.s
        out = torch.zeros(n, self.out_channels, h_, w_)
        x = pad(x, [self.k // 2] * 4)
        for i in range(h_):
            for j in range(w_):
                i_, j_ = i * self.s, j * self.s
                out[:, :, i, j] = torch.einsum('bcij,ocij->bo', x[:, :, i_:i_+self.k, j_:j_+self.k], self.weight)
        out += self.bias.view(1, -1, 1, 1)
        return out

class ConvTranspose2dManual(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2) -> None:
        super(ConvTranspose2dManual, self).__init__()
        self.s = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.rand(in_channels, out_channels, self.s, self.s))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        h_, w_ = h * self.s, w * self.s
        out = torch.zeros(n, self.out_channels, h_, w_)
        for i in range(h):
            for j in range(w):
                i_, j_ = i * self.s, j * self.s
                out[:, :, i_:i_+self.s, j_:j_+self.s] = torch.einsum('bc,coij->boij', x[:, :, i, j], self.weight)
        out += self.bias.view(1, -1, 1, 1)
        return out
