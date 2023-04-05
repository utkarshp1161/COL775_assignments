"""
refernces:
        1) https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

"""

import torch 
import torch.nn as nn


class Rnn(nn.Module):
    """
    Creates Resnet model from scratch: 
    Layers:

    
    """

    def __init__(self, n, r):
        self.num_layers = 6*n + 2
        self.num_classes = r
        self.layer_32_cross_32 = [nn.Conv2d(32, 32, 16) for i in range(2*n + 1)]
        self.layer_16_cross_16 = [nn.Conv2d(16, 16, 32) for i in range(2*n)]
        self.layer_8_cross_8 = [nn.Conv2d(8, 8, 64) for i in range(2*n)]
        self.classification_layer = nn.fullyconnected

    def forward(self, x):
        