#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:54:30 2020

@author: darp_lord
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
from sklearn.metrics import classification_report

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

N_EPOCHS=10
B_SIZE=64
LR=0.001
N_CLASSES=10

mean=np.array([0.5])
std=np.array([0.3])

data_transform=transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
train=FashionMNIST(root="../Data",
                   train=True, 
                   download=True, 
                   transform=transforms.ToTensor())

test=FashionMNIST(root="../Data",
                  train=False,
                  download=False,
                  transform=transforms.ToTensor())

train_loader=DataLoader(train, batch_size=B_SIZE, shuffle=True, num_workers=1)
test_loader=DataLoader(test, batch_size=B_SIZE, shuffle=False, num_workers=1)

classes=[
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"]


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1=nn.Conv2d(1,8,(4,4))
        self.pool1=nn.MaxPool2d((3,3), stride=2)
        self.conv2=nn.Conv2d(8,16,(3,3))
        self.pool2=nn.MaxPool2d((2,2), stride=2)
        self.flatten=nn.Flatten()
        self.l1=nn.Linear(16*5*5,256)
        self.dropout=nn.Dropout(0.3)
        self.l2=nn.Linear(256,128)
        self.lo=nn.Linear(128,N_CLASSES)
    def forward(self, x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.flatten(x)
        x=self.dropout(F.relu(self.l1(x)))
        x=self.dropout(F.relu(self.l2(x)))
        x=self.lo(x)
        return x

class GitConv(nn.Module):
    def __init__(self):
        super(GitConv, self).__init__()
        self.conv1=nn.Conv2d(1,32,(5,5), padding=2)
        self.pool=nn.MaxPool2d((2,2), stride=2)
        self.conv2=nn.Conv2d(32,64,(5,5), padding=2)
        self.flatten=nn.Flatten()
        self.l1=nn.Linear(64*7*7,1024)
        self.dropout=nn.Dropout(0.4)
        self.lo=nn.Linear(1024,N_CLASSES)
    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.flatten(x)
        x=self.dropout(F.relu(self.l1(x)))
        x=self.lo(x)
        return x

model = GitConv().to(device)
model.load_state_dict(torch.load("./TFConv10.pth"))
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)



for epoch in range(N_EPOCHS):
    running_loss=0.0
    loop=tqdm(train_loader, position=0, miniters=100)
    for counter, (images, labels) in enumerate(loop, start=1):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+= loss.detach().item()*images.size(0)
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        loop.set_postfix(loss=(running_loss/(counter*train_loader.batch_size)))

print('Finished Training..')
PATH = './TFConv20.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    y_pred=[]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, preds = torch.max(outputs, 1)
        y_pred+=preds.tolist()
    print(classification_report(test.targets, y_pred, target_names=classes))
