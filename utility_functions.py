import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from  torch.optim import SGD, lr_scheduler
from timeit import default_timer as timer
import os

####################################### CHECKPOINT #######################################
def save_model(model, history:list, model_name:str):
  '''
  This function saves the model and its training history

  Args:
  ====
  model: a Neural Network model that contains `epochs` field. an `optimizer`, and a `scheduler`
  history: a list of tuples (train_loss, val_loss, train_acc, val_acc)
  model name: the name of the model, used as file name when saving the checkpoint

  Return
  ====
  None
  '''
  os.makedirs('./model-checkpoints', exist_ok=True)

  # create a checkpoint path to save model
  path = f'./model-checkpoints/{model_name}.pt'

  # create a checkpoint object
  checkpoint = {}

  # save model and its state
  checkpoint['model'] = model
  checkpoint['state_dict']=  model.state_dict()

  # save model training steps
  checkpoint['epochs'] =  model.epochs

  # save model optimizr
  checkpoint['optimizer'] = model.optimizer
  checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()
  
  # save model scheduler
  checkpoint['scheduler'] = model.scheduler
  checkpoint['scheduler_state_dict'] =  model.scheduler.state_dict()

  # save history of loss and accuracy
  checkpoint['history'] = history

  torch.save(checkpoint, path)

def load_checkpoint(model_name:str, device):
  '''
  This function loads a checkpoint

  Args
  ====
  model_name: the name of the file

  Returns
  =======
  model: the model attached with its `epochs` field, `optimizer`, and `scheduler`
  history: a list of tuples (train_loss, val_lost, train_acc, val_acc)
  '''
  
  # path to load
  path = f'./model-checkpoints/{model_name}.pt'

  # load checkpoint
  checkpoint = torch.load(path, map_location=device)

  # load model and its state
  model = checkpoint['model']
  model.load_state_dict(checkpoint['state_dict'])

  # load number of trained epochs
  model.epochs = checkpoint['epochs']
  
  # load optimizer
  optimizer = checkpoint['optimizer']
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  model.optimizer = optimizer

  # load scheduler
  scheduler = checkpoint['scheduler']
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  model.scheduler = scheduler

  history = checkpoint['history']

  return model, history
####################################### TRANSFORMS #####################################
def subtractFromImageMean(imgTensor: torch.Tensor):
  mean = imgTensor.flatten().mean()
  return imgTensor-mean  
  

####################################### DATASETS #######################################


def MNIST_loaders(saved_path:str='./datasets', IMAGE_SIZE = 20):
  '''
  This function returns data loaders for the MNIST dataset, 
  Using batch_size=128 and transformations according to the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf

  Return
  ======
  train_loader (DataLoader): a dataloader using MNIST training set
  test_loader (DataLoader): a dataloader using MNIST testing set
  '''
  os.makedirs(saved_path, exist_ok=True)

  BATCH_SIZE = 128

  # transformations
  transformations = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, ), std=(1, ))
  ])

  # download 
  train_set = datasets.MNIST(saved_path, train=True, transform=transformations, download=True)
  test_set = datasets.MNIST(saved_path, train=False, transform=transformations, download=True)

  # split training set to two loaders
  train_loader =  DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

  # test loader 
  test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

  return train_loader, test_loader


def MNIST_augment_loaders(saved_path:str='./datasets', IMAGE_SIZE=20):
  '''
  This function returns data loaders for the MNIST dataset, 
  Using batch_size=128 and data augmentation according to the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf

  Return
  ======
  train_loader (DataLoader): a dataloader using MNIST training set
  test_loader (DataLoader): a dataloader using MNIST testing set
  '''
  os.makedirs(saved_path, exist_ok=True)

  BATCH_SIZE = 128

  # transformations
  transformations = transforms.Compose([
    transforms.RandomCrop(size=(24, 24)), # cropped 24 × 24 images from random locations
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # rescale
    transforms.RandomRotation(degrees=(-180, 180)), # rotate
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, ), std=(1, ))
  ])

  # download 
  train_set = datasets.MNIST(saved_path, train=True, transform=transformations, download=True)
  test_set = datasets.MNIST(saved_path, train=False, transform=transformations, download=True)

  # split training set to two loaders
  train_loader =  DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

  # test loader 
  test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

  return train_loader,  test_loader

def MNIST_loaders_originals(saved_path:str='./sample_data', IMAGE_SIZE=20):
  '''
  This function returns data loaders for the MNIST dataset, 
  Using batch_size=128 and transformations according to the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf
  No augmentation applied
  Original image of size 28x28 returned
  Subtract from mean
  
  Return
  ======
  train_loader (DataLoader): a dataloader using MNIST training set
  test_loader (DataLoader): a dataloader using MNIST testing set
  '''
  BATCH_SIZE = 128

  # transformations
  transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(subtractFromImageMean)
  ])

  # download 
  train_set = datasets.MNIST(saved_path, train=True, transform=transformations, download=True)
  test_set = datasets.MNIST(saved_path, train=False, transform=transformations, download=True)

  # split training set to two loaders
  train_loader =  DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

  # test loader 
  test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

  return train_loader,  test_loader

####################################### TRAIN FUNCTION #######################################

def train(model,
          criterion, 
          train_loader:DataLoader, 
          val_loader:DataLoader,
          epochs:int,
          history:list,
          model_name:str,
          device,
          save_every=20,
          print_every:int=2):
  '''
  This function trains a neural network. 

  Args:
  =====
  model: a model to train. It must be attached with a `scheduler` and an `optimizer`. Calling `model(x)` produces yhat
  criterion: a loss function
  train_loader: a DataLoader for training
  val_loader: a DataLoader for validation
  epochs: total training epochs
  model_name: used as filename when saving checkpoint
  save_every: save model after `save_every` number of epochs
  print_every: print the train/val loss  and train/val accuracy after every `print_every` number of epochs

  Return:
  ======
  None
  '''
  model.to(device)

  # start time for training
  overall_start = timer()

  try:
    print(f"Model  has been trained for {model.epochs} epochs.")
    start_epoch = model.epochs+1
  except:
    print("Start training from scratch")
    start_epoch=0
    model.epochs = 0

  for epoch in range(start_epoch,epochs):
    # track running loss and running corrects for each epoch
    train_running_loss = 0
    val_running_loss = 0
    train_running_corrects = 0
    val_running_corrects = 0
    train_size = 0.0
    val_size = 0.0

    # start an epoch with a training loop
    model.train()

    for i, (x,y) in enumerate(train_loader):
        # move data to correct device
        x = x.to(device)
        y = y.to(device)

        ## TRAIN WITH A BATCH ##
        # clear gradients
        model.optimizer.zero_grad()

        # softmax probabilities
        output = model(x)

        # compute loss
        loss = criterion(output, y)

        # compute gradient using back propagation
        loss.backward()

        # update weights using optimizer
        model.optimizer.step()
        ## FINISH TRAIN ##

        # update running loss and running corrects
        # here loss.item() is the average loss, thus multiply by the batch size to compute the running loss for the entire batch
        train_running_loss += loss.item()*x.size(0) 
        _, preds = output.max(1)
        train_running_corrects += (preds==y).sum()
        train_size += x.size(0) 
    else:
      model.epochs += 1

      # next is a validation loop
      model.eval()

      with torch.no_grad():
        for x,y in val_loader:
          # move data to correct device
          x = x.to(device)
          y = y.to(device)

          # make predictions on the validation dataset
          output = model(x)
          loss = criterion(output, y)

          # update running loss and running corrects
          val_running_loss += loss.item()*x.size(0)
          _, preds = output.max(1)
          val_running_corrects += (preds==y).sum()
          val_size += x.size(0) 
  
    # we have finished an epoch

    # call scheduler update lr if needed
    model.scheduler.step()

    # compute train/val loss and accuracy rates for the epoch
    train_loss = train_running_loss/train_size
    val_loss = val_running_loss/val_size
    train_acc = train_running_corrects/train_size
    val_acc = val_running_corrects/val_size
    history.append((train_loss, val_loss, train_acc, val_acc))
  
    # print current training history
    if (epoch%print_every==0):
      print(f"FINISH EPOCH {epoch}, training loss={train_loss:.2f}, validation loss={val_loss:.2f}")
      print(f"\t Training accuracy={train_acc*100.0:.2f}%, validation accuracy={val_acc*100.0:.2f}%")
      print(f"\t Learning rate for weights=", model.optimizer.param_groups[0]["lr"])
      if (len(model.optimizer.param_groups)>1):
        print(f"\t Learning rate for bias=", model.optimizer.param_groups[1]["lr"])

    # save model 
    if ((epoch+1)%save_every==0 or epoch==epochs-1):
      print(f"Saving {model_name}.pt ...")
      save_model(model, history, model_name)
  
  print(f"FINISH TRAINING, time elapsed: {timer()-overall_start}s")

####################################### LEARNING RATES SCHEDULERS ######################################

def LR_LAMBDA_600_400_20_schedule(epoch):
  '''
  This function returns a multiplicative factor given an integer parameter epoch 
  Using 600-400-20 schedule defined in the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf

  Use this to update the weights' learning rate
  '''
  if epoch<600:
    return 1
  elif epoch<1000:
    return 0.5
  elif epoch<1400:
    return 0.1
  elif epoch<1420:
    return 0.05
  elif epoch<1440:
    return 0.01
  elif epoch<1460:
    return 0.005
  else:
    return 0.001

def LR_LAMBDA_600_400_20_schedule_2x(epoch):
  '''
  This function returns a multiplicative factor given an integer parameter epoch 
  Using 600-400-20 schedule with double learning rate defined in the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf

  Use this to update the bias's learning_rate
  '''
  if epoch<600:
    return 2*1
  elif epoch<1000:
    return 2*0.5
  elif epoch<1400:
    return 2*0.1
  elif epoch<1420:
    return 2*0.05
  elif epoch<1440:
    return 2*0.01
  elif epoch<1460:
    return 2*0.005
  else:
    return 2*0.001
  
def LR_LAMBDA_700_200_100_schedule(epoch):
  '''
  This function returns a multiplicative factor given an integer parameter epoch 
  Using 700-200-100 schedule defined in the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf
  '''
  if epoch<700:
    return 1
  elif epoch<900:
    return 0.5
  elif epoch<1100:
    return 0.1
  elif epoch<1200:
    return 0.05
  elif epoch<1300:
    return 0.01
  elif epoch<1400:
    return 0.005
  else:
    return 0.001

def LR_LAMBDA_700_200_100_schedule_2x(epoch):
  '''
  This function returns a multiplicative factor given an integer parameter epoch 
  Using 700-200-100 schedule with double learning rate defined in the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf

  Use this to update the bias's learning_rate
  '''
  if epoch<700:
    return 2*1
  elif epoch<900:
    return 2*0.5
  elif epoch<1100:
    return 2*0.1
  elif epoch<1200:
    return 2*0.05
  elif epoch<1300:
    return 2*0.01
  elif epoch<1400:
    return 2*0.005
  else:
    return 2*0.001
  

####################################### OPTIMIZERS ######################################
def mnist_fc_optimizer(model):
  '''
  This function returns an `optimizer` and a `scheduler` used by DropConnect paper to train a fully connected NN on the MNIST dataset
  According to the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf

  Args:
  =====
  model: a fully connected model to train the MNIST dataset

  Returns:
  =======
  optimizer: a SGD optimizer with  `Initial Learning Rate`=0.1 and `momentum`=0.9
  scheduler: a lr scheduler using `LR_LAMBDA_600_400_200_schedule` to update weights' learning rate and `LR_LAMBDA_600_400_200_schedule_2x` to update bias's learning rate
  epochs: the total number of epochs to train
  '''
  iterator = model.parameters()
  weights_params = next(iterator)
  bias_params = next(iterator)

  optimizer = SGD([{'params': weights_params}, {'params': bias_params}], lr=0.1, momentum=0.9)
  scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[LR_LAMBDA_600_400_20_schedule, LR_LAMBDA_600_400_20_schedule_2x])
  epochs = 600+400*2+20*4

  return optimizer, scheduler, epochs  

def mnist_cnn_optimizer(model):
  '''
  This function returns an `optimizer` and a `scheduler` used by DropConnect paper to train a convolutional NN on the MNIST dataset
  According to the DropConnect paper http://proceedings.mlr.press/v28/wan13.pdf

  Args:
  =====
  model: a cnn model to train the MNIST dataset

  Returns:
  =======
  optimizer: a SGD optimizer with  `Initial Learning Rate`= 0.01 
  scheduler: a lr scheduler using `LR_LAMBDA_700_200_100_schedule` to update weights' learning rate and `LR_LAMBDA_700_200_100_schedule_2x` to update bias's learning rate
  epochs: the total number of epochs to train
  '''
  iterator = model.parameters()
  weights_params = next(iterator)
  bias_params = next(iterator)

  optimizer = SGD([{'params': weights_params}, {'params': bias_params}], lr=0.01)
  scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[LR_LAMBDA_700_200_100_schedule, LR_LAMBDA_700_200_100_schedule_2x])
  epochs = 700+200*2+100*4

  return optimizer, scheduler, epochs
