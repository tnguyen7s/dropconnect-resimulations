import torch
import torch.nn as nn
import utility_functions
from FourLayerDropoutConvNet import FourLayerDropoutConvNet
from utility_functions import load_checkpoint

################# DEVICE ####################
device = torch.device('cpu') # cpu by default

GPU = torch.cuda.is_available()
if GPU:
  device = torch.device('cuda') # use gpu if available
  print('Using GPU')


################# DATA ####################
train_loader, test_loader = utility_functions.MNIST_loaders()

################# TRAIN different dropout rate ##################
dropoutRates = [0.1, 0.2, 0.3, 0.4, 0.5]

for dropoutR in dropoutRates:
  # model
  model = FourLayerDropoutConvNet(dropoutR)

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
  utility_functions.train(model, criterion, train_loader, test_loader, epochs, history, f'mnist-cnn-dropout-0{dropoutR*10}-noaug', device)

