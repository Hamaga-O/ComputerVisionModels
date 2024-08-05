import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
#import dataloader as DataLoader


import argparse
import time
'''
import copy
#from scipy.stats import rice 
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
'''

cuda = False
kwargs = {'num_workers':1, 'pin_memory': True} if cuda else {}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_data = datasets.FashionMNIST('data', train = True, download = True, transform = transforms.Compose([transforms.ToTensor(),]))
train_loader = DataLoader(train_data, batch_size = 128, shuffle = False, **kwargs)

print("Training set: {} images".format(len(train_data)))

train_data, validation_data = random_split(train_data,(50000,10000))

print("Training set: {} images".format(len(validation_data)))

validation_loader = DataLoader(validation_data, batch_size = 128, shuffle = True)


#data_inputs,data_labels = next(iter(train_loader))

#print("Data inputs ", data_inputs.shape, data_inputs)
#print("Data labels ", data_labels.shape, data_labels)

#input_shape = x_train.shape[]
'''
#Normalize data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state=42)
normalizer = preprocessing.Normalizer()
normalized_train_X = normalizer.fit_transform(X_train)
normalized_test_X = normalizer.transform(X_test)
'''
class NetBlock(nn.Module):
    def __init__(self, input_channles, output_channels, identity_downsample = None, stride = 1):
        super(NetBlock,self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(input_channles, output_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, stride = stride, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv3 = nn.Conv2d(output_channels, output_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(output_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

        '''
        super().__init__()
        if not subsample:
            output_channels = input_channles
        '''
    def forward(self,x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)


        if self.identity_downsample != None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x 
#initialize NetBlock
#myNetBlock = NetBlock(input_channles = 128, output_channels = 128)
#print(myNetBlock)

#show training data params
#for name, param in myNetBlock.named_parameters():
 #   print(f"Parameter{name}, shape{param.shape}")

#test for gpu
#gpu_avail = torch.cuda.is_available()
#print("got gpu? ", gpu_avail)

#loss ting
#loss_module = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.SGD(myNetBlock.parameters(), lr=0.1)


#train_model(model, optimizer, train_loader, loss_module)

class ResNet(nn.Module):
    def __init__(self, NetBlock, layers, image_channels, num_classes):
        super(ResNet,self).__init__()
        self.input_channles = 14
        self.conv1 = nn.Conv2d(image_channels, 14, kernel_size = 7, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(14)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        #ResNet Layers
        self.layer1 = self.make_layer(NetBlock, layers[0], output_channels = 14, stride = 1)
        self.layer2 = self.make_layer(NetBlock, layers[0], output_channels = 28, stride = 2)
        self.layer3 = self.make_layer(NetBlock, layers[0], output_channels = 56, stride = 2)
        self.layer4 = self.make_layer(NetBlock, layers[0], output_channels = 112, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(112*4, num_classes)

    def forward(self,x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def make_layer(self,NetBlock,num_residual_NetBlocks, output_channels, stride):
        #Build layer of residual blocks
        identity_downsample = None
        layers = []

        if stride != 1 or self.input_channles != output_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.input_channles, output_channels * 4, kernel_size = 1, stride = stride), nn.BatchNorm2d(output_channels*4))

        #change number of channels using layer
        layers.append(NetBlock(self.input_channles,output_channels, identity_downsample, stride)) 
        self.input_channles = output_channels * 4 #14 * 4 should be 256 channels

        for i in range(num_residual_NetBlocks-1):
            layers.append(NetBlock(self.input_channles, output_channels)) # outchannles are 14 from 256 then 
            #converted back to 256 as 14*4 
        
        #return layer of blocks
        return nn.Sequential(*layers)


def ResNet50(image_channels = 1, num_classes = 1000):
    return ResNet(NetBlock, [3,4,6,3], image_channels, num_classes)
    
model = ResNet50()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)
def train_model():
    EPOCHS = 15
    train_samples = 50000
    validation_samples = 10000
    train_costs = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):

        train_running_loss = 0
        correct_train = 0

        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

            _, pred_outputs = torch.max(pred.data, 1)

            correct_train += (pred_outputs == labels).float().sum().item()

            train_running_loss += (loss.data.item() * inputs.shape[0])

    train_epoch_loss = train_running_loss / train_samples_num
    train_acc = correct_train/train_samples_num

    info = "[Epoch {}/{}]: train_loss = {:0.6f} | train accuracy = {:0.3f}"
    print(info.format(epoch+1,EPOCHS, train_epoch_loss, train_acc))
'''
def test():
    x = torch.randn(2,3,224,224)
    y = NetBlock(x).to(device)
    print (y.shape)

test()
'''
train_model()
