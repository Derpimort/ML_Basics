#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:58:31 2020

@author: darp_lord
"""

import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'



# VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version $VERSION



# Result Visualization Helper
from matplotlib import pyplot as plt

M, N = 4, 6
RESULT_IMG_PATH = '/tmp/test_result.jpg'
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_results(images, labels, preds):
  images, labels, preds = images[:M*N], labels[:M*N], preds[:M*N]
  inv_norm = transforms.Normalize(
      mean=(-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
      std=(1/0.2023, 1/0.1994, 1/0.2010))

  num_images = images.shape[0]
  fig, axes = plt.subplots(M, N, figsize=(16, 9))
  fig.suptitle('Correct / Predicted Labels (Red text for incorrect ones)')

  for i, ax in enumerate(fig.axes):
    ax.axis('off')
    if i >= num_images:
      continue
    img, label, prediction = images[i], labels[i], preds[i]
    img = inv_norm(img)
    img = img.permute(1, 2, 0) # (C, M, N) -> (M, N, C)
    label, prediction = label.item(), prediction.item()
    if label == prediction:
      ax.set_title(u'\u2713', color='blue', fontsize=22)
    else:
      ax.set_title(
          'X {}/{}'.format(CIFAR10_LABELS[label],
                          CIFAR10_LABELS[prediction]), color='red')
    ax.imshow(img)
  plt.savefig(RESULT_IMG_PATH, transparent=True)



# Define Parameters
FLAGS = {}
FLAGS['data_dir'] = "/tmp/fashionmnist"
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.1
FLAGS['momentum'] = 0.9
FLAGS['num_epochs'] = 300
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = True
FLAGS['num_classes']=10



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
from torch.utils.data.distributed import DistributedSampler
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.core.xla_model import RateTracker
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

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


def train():
    torch.manual_seed(1)
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
    
    train_dataset=FashionMNIST(root=os.path.join(FLAGS['data_dir'], str(xm.get_ordinal())),
                       train=True, 
                       download=True, 
                       transform=transform_train)
    
    
    
    test_dataset=FashionMNIST(root=os.path.join(FLAGS['data_dir'], str(xm.get_ordinal())),
                      train=False,
                      download=False,
                      transform=transform_test)
    
    train_sampler = DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    
    train_loader = DataLoader(
          train_dataset,
          batch_size=FLAGS['batch_size'],
          sampler=train_sampler,
          num_workers=FLAGS['num_workers'],
          drop_last=True)
      
    test_loader = DataLoader(
          test_dataset,
          batch_size=FLAGS['batch_size'],
          shuffle=False,
          num_workers=FLAGS['num_workers'],
          drop_last=True)
    
    
    learning_rate = FLAGS['learning_rate'] * xm.xrt_world_size()
    
    device = xm.xla_device()
    
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
        
    model= WideResNet(num_classes=FLAGS['num_classes'], depth=28, widen_factor=10).to(device)
    
    loss_func=nn.CrossEntropyLoss()
    optimizer=SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler=lr_scheduler.MultiStepLR(optimizer, milestones=[150,225], gamma=0.1)
    
    def train_loop_fn(loader):
        tracker=RateTracker()
        model.train()
        print("Start Training")
        for counter, (images, labels) in enumerate(train_loader, start=1):
            outputs = model(images)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS['batch_size'])
            if counter % FLAGS['log_steps'] == 0:
                print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                xm.get_ordinal(), counter, loss.item(), tracker.rate(),
                tracker.global_rate(), time.asctime()), flush=True)
            scheduler.step()
    
    def test_loop_fn(loader):
        total_samples = 0
        correct = 0
        model.eval()
        data, pred, target = None, None, None
        for data, target in loader:
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size()[0]
    
        accuracy = 100.0 * correct / total_samples
        print('[xla:{}] Accuracy={:.2f}%'.format(
            xm.get_ordinal(), accuracy), flush=True)
        return accuracy, data, pred, target
    
        # print(f'Epoch [{epoch+1}/{N_EPOCHS}] Loss= {(running_loss/counter)}')
        # if((epoch+1)%125==0):
        #     model.eval()
        #     with torch.no_grad():
        #         y_pred=[]
        #         for images, labels in test_loader:
        #             images = images.to(device)
        #             labels = labels.to(device)
        #             outputs = model(images)
        #             # max returns (value ,index)
        #             _, preds = torch.max(outputs, 1)
        #             y_pred+=preds.tolist()
        #         print(classification_report(test.targets, y_pred, target_names=classes))
        #     model.train()
        #     torch.save(model.state_dict(), DATA_DIR+"/WRN-28-10_%d"%(epoch+1))
        # scheduler.step()
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(1, FLAGS['num_epochs'] + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))
          
        para_loader = pl.ParallelLoader(test_loader, [device])
        accuracy, data, pred, target  = test_loop_fn(para_loader.per_device_loader(device))
        if FLAGS['metrics_debug']:
            xm.master_print(met.metrics_report(), flush=True)
        scheduler.step()
    PATH = FLAGS['data_dir']+"/WRN-28-10_F.pth"
    torch.save(model.state_dict(), PATH)
    return accuracy, data, pred, target


# Start training processes
def _mp_fn(rank, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    accuracy, data, pred, target = train()
    if rank == 0:
       # Retrieve tensors that are on TPU core 0 and plot.
       plot_results(data.cpu(), pred.cpu(), target.cpu())

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS['num_cores'],
          start_method='fork')


from google.colab.patches import cv2_imshow
import cv2
img = cv2.imread(RESULT_IMG_PATH, cv2.IMREAD_UNCHANGED)
cv2_imshow(img)


print('Finished Training..')


