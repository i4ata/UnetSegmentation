"""
Test that my implementations of the
1. Convolutional layer
2. Transposed convolutional layer
3. MaxPool layer
are mathematically sound (i.e. they produce the same results as built-in layers from torch)
"""

import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d
import unittest
from layers import Conv2dManual, ConvTranspose2dManual

class TestConvolution(unittest.TestCase):

    def test_convolution(self):
        
        x = torch.rand(5, 3, 200, 200)
        my_layer = Conv2dManual(3, 5, 3, 1)
        torch_layer = nn.Conv2d(3, 5, 3, 1, padding='same')
        my_layer.load_state_dict(torch_layer.state_dict())
        self.assertTrue(
            torch.isclose(my_layer(x), torch_layer(x), atol=1e-5).all().item()
        )

    def test_transposed_convolution(self):

        x = torch.rand(5, 5, 200, 200)
        my_layer = ConvTranspose2dManual(5, 3, 2)
        torch_layer = nn.ConvTranspose2d(5, 3, 2, 2)
        my_layer.load_state_dict(torch_layer.state_dict())
        self.assertTrue(
            torch.isclose(my_layer(x), torch_layer(x), atol=1e-5).all().item()
        )

    def test_max_pool(self):

        x = torch.rand(5, 3, 200, 200)
        n, c, h, w = x.shape
        my_y = x.reshape(n, c, h // 2, 2, w // 2, 2).amax((3, 5))
        torch_y = max_pool2d(x, kernel_size=2, stride=2)
        self.assertTrue(torch.equal(my_y, torch_y))
