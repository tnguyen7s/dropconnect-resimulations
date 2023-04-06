import torch.nn as nn
from torch.nn import functional as F

import torch.nn as nn
from torch.nn import functional as F
class FourLayerNoDropoutConvNet(nn.Module):
  def __init__(self):
    super().__init__()

    # 1st 
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1, padding=0)
    nn.init.kaiming_normal_(self.conv1.weight)
    self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    # 2nd
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=1, padding=0)
    nn.init.kaiming_normal_(self.conv2.weight)
    self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
    
    self.flatten = nn.Flatten()
    # 3rd 
    self.fc1 = nn.Linear(in_features=256, out_features=150)
    nn.init.kaiming_normal_(self.fc1.weight)

    #4th
    self.fc2 = nn.Linear(in_features=150, out_features=10)
    nn.init.kaiming_normal_(self.fc2.weight)

    self.softmax = nn.Softmax(1)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool2(x)

    x = self.flatten(x)
    x = self.fc1(x)
    x = F.relu(x)

    scores = self.fc2(x)
    props = self.softmax(scores)
    return props