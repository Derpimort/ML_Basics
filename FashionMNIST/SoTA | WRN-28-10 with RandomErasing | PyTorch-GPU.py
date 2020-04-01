#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:57:41 2020

@author: darp_lord
"""


import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD
from torch.utils.data import Subset, Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import classification_report

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

DATA_DIR="/kaggle/working/"
B_SIZE=625
LR=0.1
N_EPOCHS=300
N_CLASSES=10


transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p = 0.5, scale = (0.02,0.4), ratio = (0.3,3.3), value=0.4914),
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

train=FashionMNIST(root=DATA_DIR,
                   train=True, 
                   download=True, 
                   transform=transform_train)



test=FashionMNIST(root=DATA_DIR,
                  train=False,
                  download=False,
                  transform=transform_test)

train_loader=DataLoader(train, batch_size=B_SIZE, shuffle=True, num_workers=2)
test_loader=DataLoader(test, batch_size=B_SIZE, shuffle=False, num_workers=2)

# examples=iter(train_loader)
# samples,labels=examples.next()
# print(samples.shape, labels.shape)
# plt.figure(figsize=(24,16))
# for i in range(6):
# 	plt.subplot(2,3,i+1)
# 	plt.imshow(samples[i][0], cmap='gray')
# plt.show()

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = xm.xla_device()
print(device)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
model= WideResNet(num_classes=N_CLASSES, depth=28, widen_factor=10).to(device)

loss_func=nn.CrossEntropyLoss()
optimizer=SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler=lr_scheduler.MultiStepLR(optimizer, milestones=[150,225], gamma=0.1)


print("Start Training")
for epoch in range(N_EPOCHS):
    running_loss=0.0
#     loop=tqdm(train_loader)
    
    for counter, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+= loss.detach().item()
#         loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
#         loop.set_postfix(loss=(running_loss/counter))
    print(f'Epoch [{epoch+1}/{N_EPOCHS}] Loss= {(running_loss/counter)}')
    if((epoch+1)%125==0):
        model.eval()
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
        model.train()
        torch.save(model.state_dict(), DATA_DIR+"/WRN-28-10_%d"%(epoch+1))
    scheduler.step()
    
print('Finished Training..')
PATH = DATA_DIR+"/WRN-28-10_F.pth"
torch.save(model.state_dict(), PATH)

