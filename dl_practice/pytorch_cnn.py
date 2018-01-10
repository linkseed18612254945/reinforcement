import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

batch_size = 32

train_data = datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self, n_input, n_output):
        super(CNN, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
