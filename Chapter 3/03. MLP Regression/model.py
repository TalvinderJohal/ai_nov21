import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary

# YOUR CODE HERE

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 100)
        self.layer4 = nn.Linear(100, 40)
        self.layer5 = nn.Linear(40, 1)
        

    def forward(self, x):
        x = self.layer(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        return x


# mlp = MLP()
# print(mlp)