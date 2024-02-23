"""
_summary_
train and test traffic sign dataset    
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import SignDataset
import networks


class Trainer:
    def __init__(self, options):
        self.opt = options
        # self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        
        # load dataset
        self.dataset_path = self.opt.data_path
        self.batch_size = self.opt.batch_size
        self.dataset = SignDataset.SignDataLoader(self.dataset_path, self.batch_size)
        
        self.train_loader = self.dataset.getTrainLoader()
        self.valid_loader = self.dataset.getValidLoader()
        self.test_loader = self.dataset.getTestLoader()
        
        # loss function and optim
        
        # model
        self.num_classes = self.opt.num_classes
        if(self.opt.model == "LeNet"):
            self.model = networks.LeNet(self.num_classes)
        elif(self.opt.model == "AlexNet"):
            self.model = networks.AlexNet(self.num_classes)
        elif(self.opt.model == "VGG11" or self.opt.model == "VGG13" or self.opt.model == "VGG16" or self.opt.model == "VGG19"):
            self.model = networks.VGG(self.opt.model, self.num_classes)
        elif(self.opt.model == "ResNet"):
            self.model = networks.ResNet([2, 2, 2, 2],self.num_classes)
        else :
            # default model
            self.model = networks.LeNet(self.num_classes)
        print("-----")
        print(self.model)
        print("-----")
        
        # other params
        self.num_epochs = self.opt.num_epochs
    
    
    def train(self):
        # loss and optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        print('Training Begin!')
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()  # 设置为训练模式
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # 在每个epoch结束后评估模型在验证集上的性能
            self.model.eval()  # 设置为评估模式
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
            
            accuracy = total_correct / total_samples
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Validation Accuracy: {accuracy:.4f}')

        print('Training Finished!')
    
    def test(self):
        print('Test Begin!')
        self.model.eval()  # 设置为评估模式
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)               # 找到类别
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f'Test Accuracy: {accuracy:.4f}')
