import torch.nn as nn

class TwoLayersFcDropoutNet(nn.Module):
  '''
  This is a Two Layers Fully Connected with Standard Dropout Neural Net (dropout probability=0.5)
  '''
  def __init__(self, in_size, out_size, p=0.4):
    super().__init__()

    self.flatten = nn.Flatten()

    # layer 1
    self.layer1 = nn.Linear(in_features=in_size, out_features=800)
    nn.init.normal_(self.layer1.weight, mean=0, std=0.1)
    nn.init.normal_(self.layer1.bias, mean=0, std=0.1)
    self.drop1 = nn.Dropout(p)
    self.relu1 = nn.ReLU()

    # layer 2
    self.layer2 = nn.Linear(in_features=800, out_features=800)
    nn.init.normal_(self.layer2.weight, mean=0, std=0.1)
    nn.init.normal_(self.layer2.bias, mean=0, std=0.1)
    self.drop2 = nn.Dropout(p)
    self.relu2 = nn.ReLU()

    # output layer
    self.out = nn.Linear(in_features=800, out_features=out_size)
    nn.init.normal_(self.out.weight, mean=0, std=0.1)
    nn.init.normal_(self.out.bias, mean=0, std=0.1)
    self.softmax = nn.Softmax(1)

  def forward(self, x):
    x = self.flatten(x)

    x = self.layer1(x)
    x = self.drop1(x)
    x = self.relu1(x)

    x = self.layer2(x)
    x = self.drop2(x)
    x = self.relu2(x)

    x = self.out(x)
    props = self.softmax(x)

    return props