import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

use_gpu = False
batch_size = 64
lr = 1e-3
num_epoch = 10

img_size = 28 * 28
hidden1 = 128
hidden2 = 128
output_size = 10


class NeuralNetwork(nn.Module):
    def __init__(self, num_input, hidden_size_1, hidden_size_2, num_output):
        super(NeuralNetwork, self).__init__()
        self.num_input = num_input
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_output = num_output

        self.layer1 = nn.Sequential(
            nn.Linear(num_input, hidden_size_1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size_2, output_size),
            nn.Softmax()
        )

    def forward(self, input_batch):
        x = input_batch
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.output_layer(x)
        return output

train_data = datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(),download=True)
test_data = datasets.MNIST(root='./mnist', transform=transforms.ToTensor(), train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = NeuralNetwork(img_size, hidden1, hidden2, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    print('Training......')
    time_since = time.time()
    for epoch in range(num_epoch):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            label = Variable(label)
            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            output = model.forward(img)
            _, pred = torch.max(output, dim=1)
            running_acc += (pred == label).sum().data[0]
            loss = criterion(output, label)
            running_loss += loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                time_util = time.time()
                print('Epoch[{}/{}], Loss: {:.6f}, Acc: {:.6f}, use_time: {:.2f}'
                      .format(epoch + 1, num_epoch, running_loss / i, running_acc / (batch_size * i), time_util - time_since))


def test():
    print('Testing......')
    model.eval()
    num_correct = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        output = model.forward(img)
        _, pred = torch.max(output, 1)
        num_correct += (pred == label).sum().data[0]
    print('Acc is {:.4f}'.format(num_correct / len(test_data)))

if __name__ == '__main__':
    train()
    test()