import torch
import torch.nn as nn
import utility_functions
from FourLayerNoDropoutConvNet import FourLayerNoDropoutConvNet
from utility_functions import load_checkpoint

################# DEVICE ####################
device = torch.device('cpu') # cpu by default

GPU = torch.cuda.is_available()
if GPU:
  device = torch.device('cuda') # use gpu if available
  print('Using GPU')


################# DATA ####################
train_loader, test_loader = utility_functions.MNIST_loaders()

################# TRAIN ##################
# model
model = FourLayerNoDropoutConvNet()

# optimizer & scheduler
optimizer, scheduler, epochs = utility_functions.mnist_cnn_optimizer(model)
model.optimizer = optimizer
model.scheduler = scheduler

# loss function
criterion = nn.CrossEntropyLoss()

# to save acc and loss
history = []

# get model from checkpoints
# model, history = load_checkpoint('mnist-cnn-nodropout-noaug')


# train
utility_functions.train(model, criterion, train_loader, test_loader, epochs, history, 'mnist-cnn-nodropout-noaug', device)

