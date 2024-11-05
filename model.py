import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
  def __init__(self):
      super(MyNet, self).__init__()

      self.conv1 = nn.Conv2d(in_channels=3, 
                            out_channels=32, 
                            kernel_size=3,
                            stride=1, 
                            padding=0)  # Output: 32 x 30 x 30
      
      self.conv2 = nn.Conv2d(in_channels=32, 
                            out_channels=32, 
                            kernel_size=3,
                            stride=2, 
                            padding=0)  # Output: 32 x 14 x 14
      
      # Calculate the correct input features for the linear layer
      # After conv2: 32 channels, 14x14 feature map
      self.linear = nn.Linear(in_features=32 * 14 * 14,  # 32 * 14 * 14 = 6272
                            out_features=125)
      
      self.out = nn.Linear(in_features=125,
                          out_features=2)
      
  def forward(self, x):
      # Input: batch_size x 3 x 32 x 32
      h = F.relu(self.conv1(x))        # -> batch_size x 32 x 30 x 30
      h = F.relu(self.conv2(h))        # -> batch_size x 32 x 14 x 14
      h = h.reshape(h.shape[0], -1)    # -> batch_size x (32 * 14 * 14)
      h = F.relu(self.linear(h))       # -> batch_size x 125
      y = self.out(h)                  # -> batch_size x 2
      return y