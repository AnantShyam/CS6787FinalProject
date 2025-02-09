import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F

class CIFAR10Net(nn.Module):
    
    def __init__(self):
        
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 512, bias=False)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
