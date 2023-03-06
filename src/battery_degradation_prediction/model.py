"""model module"""
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class Net(nn.Module):
    """TODO"""

    def __init__(self, input_shape, window_size):
        super().__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(window_size*8, 1)

    def forward(self, x):
        """TODO"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc4(x))
        return x