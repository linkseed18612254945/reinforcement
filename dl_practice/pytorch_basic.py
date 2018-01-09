import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def linear_function():
    X = torch.randn(3, 1)
    W = torch.randn(4, 3)
    b = torch.randn(4, 1)
    result = torch.matmul(W, X) + b
    print(type(result))
    return result


def sigmoid(z):
    z = torch.DoubleTensor(z)
    sig = F.sigmoid(z)
    return sig


def cost(logits, labels):

    input = Variable(torch.randn(1, 4))
    target = Variable(torch.LongTensor(1).random_(4))
    loss = F.cross_entropy(logits, labels)
    return loss

if __name__ == '__main__':
    logits = sigmoid([0.2, 0.4, 0.7, 0.9]).unsqueeze(0)
    labels = Variable(torch.LongTensor([2]))
    cost = cost(logits, labels)
    print(torch.ones([3, 3]))

