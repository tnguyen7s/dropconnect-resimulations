import torch.nn as nn
from torch.nn import functional as F
class DropConnectLinear(nn.Linear):
  '''
  A nn.Linear layer from torch is overriden with a new `forward` function where dropping connections is applied
  Following the implementation from the DropConnect paper [Regularization of Neural Networks using DropConnect](https://proceedings.mlr.press/v28/wan13.pdf) and this [PowerPoint](https://cds.nyu.edu/wp-content/uploads/2014/04/dropc_slides.pdf) 
  '''
  def __init__(self, in_features:int, out_features:int, p: float=0.5):
    '''
    Args
    p: is the probability the connection is dropped
    p=0.5 is used by DropConnect paper

    The class is inherited from nn.Linear
    '''
    super().__init__(in_features, out_features)
    self.p = p

  # implementation 1
  # def forward(self, input):
  #     # a mask matrix M is  drawn from a Bernoulli(p) to mask out elements of both the weight matrix and the biases
  #     # here, different mask is created for each training example
  #     # Bernoulli is a special case of binomial distribution with one trial; thus we can use binomial function from np.random to draw a mask
  #     # 1-p is the success probability or the probability of keeping the connection
  #     M = torch.Tensor(np.random.binomial(n=1, p=1-self.p, size=(self.out_features, self.in_features+1))) 

  #     # apply the mask to weights and biases
  #     weights_drop = self.weight*M[:, :self.in_features]
  #     bias_drop = self.bias*M[:, -1]

  #     # matrix multiplication to compute input to activation function using weights_drop and bias_drop
  #     return F.linear(input, weights_drop, bias_drop)

  # implementation 2: using torch.nn.functional.dropout built in function to drop elements in weights and bias
  def forward(self, input):
    weight_drop = F.dropout(self.weight, self.p, training=self.training)
    bias_drop = F.dropout(self.bias, self.p, training=self.training)

    return F.linear(input, weight_drop, bias_drop)

class TwoLayersFcDropconnectNet(nn.Module):
  '''
  This is a Two Layers Fully Connected with DropConnect Neural Net (drop connections with probability=0.5)
  '''
  def __init__(self, in_size, out_size, p=0.5):
    super().__init__()

    self.flatten = nn.Flatten()

    # layer 1
    self.layer1 = DropConnectLinear(in_features=in_size, out_features=800, p=p)
    nn.init.normal_(self.layer1.weight, mean=0, std=0.1)
    nn.init.normal_(self.layer1.bias, mean=0, std=0.1)
    self.relu1 = nn.ReLU()

    # layer 2
    self.layer2 = DropConnectLinear(in_features=800, out_features=800, p=p)
    nn.init.normal_(self.layer2.weight, mean=0, std=0.1)
    nn.init.normal_(self.layer2.bias, mean=0, std=0.1)
    self.relu2 = nn.ReLU()

    # output layer
    self.out = DropConnectLinear(in_features=800, out_features=out_size, p=p)
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