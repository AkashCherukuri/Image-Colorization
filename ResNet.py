import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 128
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''
    model.train()
    running_loss = 0
    for X, y_true in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        y_true = y_true.to(device)
        # Forward pass
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss
    
def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 1
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)

# define transforms
transforms = transforms.Compose([transforms.Resize((69, 69)),   # Nice
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

class ResNet(nn.Module):
  def __init__(self):
    super(ResNet,self).__init__()
    
    self.Block1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=16, stride=2, kernel_size=7), # Gives an output of 32x32
      nn.ReLU(),
      nn.MaxPool2d(stride=2, kernel_size=2),                              # Output is shrunk to 16x16
    )

    self.Block2 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
    )

    self.Block2_3 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=3, padding=1), # Shrunk to 8x8
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
    )

    self.Resize2_3 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=1),
      nn.ReLU(),
    )
    

    self.Block3 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
    )

    self.Block3_4 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3, padding=1), # Shrunk to 4x4
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
        nn.ReLU(),
    )

    self.Resize3_4 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=1),
      nn.ReLU(),
    )

    self.Block4 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
    )

    self.Block4_5 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3, padding=1), # Shrunk to 2x2
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
    )

    self.Resize4_5 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=1),
      nn.ReLU(),
    )

    self.Block5 = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
      nn.ReLU(),
    )

    self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
    self.LinearOut = nn.Linear(128,10)

  def forward(self, x):
    x = self.Block1(x)
    for i in range(2):
      x = self.Block2(x) + x
    x = self.Block2_3(x) + self.Resize2_3(x) 
    for i in range(2):
      x = self.Block3(x) + x
    x = self.Block3_4(x) + self.Resize3_4(x) 
    for i in range(3):
      x = self.Block4(x) + x
    x = self.Block4_5(x) + self.Resize4_5(x) 
    for i in range(2):
      x = self.Block5(x) + x
    x = self.AvgPool(x)
    x = torch.flatten(x,1)

    logits = self.LinearOut(x)
    prob = F.log_softmax(logits, dim=1)

    return logits, prob

# torch.manual_seed(RANDOM_SEED)

model = ResNet().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

torch.save(model, 'dat.pth')