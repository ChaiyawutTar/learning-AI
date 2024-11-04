import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np

class MyNet(nn.Module):
    def __init___(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Connv2D(in_channels=3, 
                                out_channels=32, 
                                kernel_size=3,
                                stride=1, 
                                padding=0)
        self.conv2 = nn.Connv2D(in_channels=32, 
                                out_channels=32, 
                                kernel_size=3,
                                stride=2, 
                                padding=0)
        
        self.linear = nn.Linear(in_features = 1568,
                                out_features = 125)
        
        self.out = nn.Linear(in_features = 125,
                            out_features = 2)
        
    def forward(self, x):
        # x shape = (batch_size, channels, H, W)
    
        h = self.conv1(x)
        h = F.relu(h)
        h = F.relu(self.conv2(x))
        # h shape = (batch_size, 32, H', W')
        # h -> (batch_size, 32 * H' * W')
        h = h.reshape(h.shape[0], -1)
        h = F.relu(self.linear(h))
        y_pred = self.out(h)
        
        return y_pred