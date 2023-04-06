import torch.nn as nn
class TwoLayersFcNoDropoutNet(nn.Module):
  '''
  This is a Two Layers Fully Connected with No Dropout Neural Net
  '''
  def __init__(self, in_size, out_size):
    '''
    Args:
    ====
    in_size: the number of neurons in the input layer
    out_size: the number of neurons in the output layer
    '''
    super().__init__()

    self.flatten = nn.Flatten()

    # layer 1
    self.layer1 = nn.Linear(in_features=in_size, out_features=800)
    nn.init.normal_(self.layer1.weight, mean=0, std=0.1)
    nn.init.normal_(self.layer1.bias, mean=0, std=0.1)
    self.relu1 = nn.ReLU()

    # layer 2
    self.layer2 = nn.Linear(in_features=800, out_features=800)
    nn.init.normal_(self.layer2.weight, mean=0, std=0.1)
    nn.init.normal_(self.layer2.bias, mean=0, std=0.1)
    self.relu2 = nn.ReLU()

    # output layer
    self.out = nn.Linear(in_features=800, out_features=out_size)
    nn.init.normal_(self.out.weight, mean=0, std=0.1)
    nn.init.normal_(self.out.bias, mean=0, std=0.1)
    self.softmax = nn.Softmax(1)

  def forward(self, x):
    x = self.flatten(x)

    x = self.layer1(x)
    x = self.relu1(x)

    x = self.layer2(x)
    x = self.relu2(x)

    x = self.out(x)
    props = self.softmax(x)

    return props