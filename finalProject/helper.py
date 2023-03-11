import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

# +
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class ReshapeTransform:
    """
    Transformation that reshapes tensor to a given `new_size`.
    """
    
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

def get_data_loaders(train_batch_size,  val_batch_size):
    """
    Return train/validation DataLoader for the MNIST dataset.
    """
    
    data_transform = Compose([ToTensor(), ReshapeTransform((-1,))]) 
    #torchvision provides datasets adapted for CNNs, so each MNIST sample is a tensor of shape (1, 28, 28) 
    #representing respectively the number of channels (1, since they are greyscale images), the height in pixels (28)
    #and the width in pixels (28). But for a fully-connected network, we want to "flatten" this tensor into a 1D tensor 
    #of size 28**2. This is accomplished by the ReshapeTransform.
    #Data offered by torchvision is already normalized in [0, 1], so no need to divide it by 255.
    
    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=train_batch_size, shuffle=True
    )
    
    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False), batch_size=val_batch_size, shuffle=False
    )
    
    return train_loader, val_loader

def supervised_loss(predicted : torch.Tensor, target : torch.Tensor, m : int = 2):
    """
    Loss function for the supervised training part (equation [12] of https://www.pnas.org/content/116/16/7723).
    """
    
    #Convert each class to a "spin-like" one-hot vector
    #e.g., if the number of classes is 5, then a sample of class 1 is represented as [-1, +1, -1, -1, -1]
    
    one_hot = target.new_full(predicted.shape, fill_value=-1, dtype=torch.int64) #Tensor of shape (batch_size, num_classes) full of -1
    one_hot.scatter_(1, target.view(-1, 1), 1) #At each row i of one_hot, insert a +1 value at position target[i]
    
    return torch.sum(torch.abs(predicted - one_hot)**m)

class Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, n=1, beta=.1):
        """Builds the neural network architecture presented in [1].

        n : exponent of activation function for the first hidden layer (n = 1 for ReLU)
        beta : parameter for the activation function in the top layer
        input_dim : input dimensionality
        output_dim : output dimensionality
        
        [1]: "Unsupervised learning by competing hidden units", D. Krotov, J. Hopfield
        """
        super(Net, self).__init__()
        
        #Store parameters
        self.n = n
        self.beta = beta
        
        #Define layers
        self.hidden = nn.Linear(input_dim, 2000)
        self.top    = nn.Linear(2000, output_dim)
        
    def forward(self, x):
        x = F.relu(self.hidden(x))**self.n
        x = torch.tanh(self.beta * self.top(x)) 
        
        return x