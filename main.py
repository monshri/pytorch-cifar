'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import PIL

import os
import argparse

from models import *
from utils import progress_bar
import logging
import time
import datetime

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
      self.std = std
      self.mean = mean
        
    def __call__(self, tensor):
      noise = torch.randn(tensor.size()) * self.std + self.mean
      res = tensor + noise
      return torch.clamp(input=res, min=-0.5, max=0.5)
    
    def __repr__(self):
      return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Training
def train(epoch):
    logging.info('Epoch: {}'.format(epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_loss /= len(trainloader)
    acc = 100.*correct/total
    logging.info('Train Loss: {}'.format(train_loss))
    logging.info('Train Accuracy: {}%'.format(acc))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        test_loss /= len(testloader)
        acc = 100.*correct/total
        logging.info('Test Loss: {}'.format(test_loss))
        logging.info('Test Accuracy: {}%'.format(acc))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logger.info('Saving..')
        torch.save({'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()}, 
        "/content/gdrive/My Drive/IDL_Project/models/Shufflenet{}".format(seed))
     
        best_acc = acc
    return test_loss, acc    


# Argument parsing 
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')    

# Train models for different seed number
# Shriti 
seeds = np.arange(42, 47)
# Kaichen
#seeds = np.arange(47,52)
# Shivani
#seeds = np.arange(52,57)
# Kriti
#seeds = np.arange(57,62)


# For logging information of training
logger = logging.getLogger("")

# reset handler
for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

# set handler
stream_hdlr = logging.StreamHandler()
file_hdlr = logging.FileHandler('/content/gdrive/My Drive/IDL_Project/logs/log_{}.log.log'.format(datetime.datetime.now()))

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
stream_hdlr.setFormatter(formatter)
file_hdlr.setFormatter(formatter)

logger.addHandler(stream_hdlr)
logger.addHandler(file_hdlr)

logger.setLevel(logging.INFO)

for seed in seeds:
    logging.info('Seed: {}'.format(seed))
    torch.manual_seed(seed)
    # Data Preparation
    transform_train = transforms.Compose([
      transforms.ColorJitter(hue=.05, saturation=.05),
      transforms.RandomCrop(32, padding=4),
      transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
      AddGaussianNoise(0.0, 0.15)
      ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        AddGaussianNoise(0.0, 0.06)
      ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Model building
    logging.info('Building model for seed: {}'.format(seed))
    net = ShuffleNetV2(2)
    net = net.to(device)
    if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True

    ##### NOT REQUIRED ###### 
    #if args.resume:
      # Load checkpoint.
      #logger.info("Resuming from Checkpoint")
      #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
      #checkpoint = torch.load('./checkpoint/ckpt.pth')
      #net.load_state_dict(checkpoint['net'])
      #best_acc = checkpoint['acc']
      #start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, verbose=True)
    for epoch in range(start_epoch, start_epoch+200):
      train(epoch)
      test(epoch)
      scheduler.step()
      
    torch.cuda.empty_cache()
    del net
    del criterion
    del optimizer
    del scheduler
    del trainloader
    del testloader
    del transform_train
    del transform_test
