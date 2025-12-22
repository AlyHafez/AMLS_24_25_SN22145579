import torch 
from torch import nn
from torch import optim

import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, recall_score, precision_score

import matplotlib.pyplot as plt
import numpy as np


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.dropout1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(128, num_classes) 

    def forward(self, x):
        """
        define forward pass for neural network
        Parameters:
        x: Input tensor

        Returns:
        torch.Tensor
        """

        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

def trainCNN(lr:float, num_epoch:int, train, val, input_dim: int, num_classes:int, seed:int):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr =lr)
    train_losses = []
    val_losses = []
    val_accuracies = [] 
    train_accuracies=[]
    period = 10
    minloss = 10.0
    bestacc = 0.0
    delta = 1e-3
    counter=0.0
    for epoch in range(num_epoch):
        model.train()
        correct = 0
        total = 0

        running_loss = 0.0
        val_running_loss=0.0
        for i, data in enumerate(train, 0):
        # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze().long()


            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs, 1)
            
            running_loss += loss.item()
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / len(train)
        epoch_train_acc = correct / total

        print(f"Epoch [{epoch + 1}/{num_epoch}]")
        
        
        print(f"loss:{running_loss}")
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        model.eval()
        val_running_loss = 0.0

        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val:
                images = images.to(device)
                labels = labels.to(device).squeeze().long()
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss+=val_loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
            
                epoch_val_loss = val_running_loss / len(val)
                epoch_val_acc = val_correct / val_total

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        if (minloss-delta)>epoch_val_loss:
            minloss = epoch_val_loss
            counter = 0 
            bestacc = epoch_val_acc

            print(f"epoch{epoch} performs better, acccuracy:{bestacc}")
        else:
            counter+=1
        if((period<=counter) or epoch>=num_epoch):
            print(f"early stopping implemented at:{epoch-period} with accuracy : {bestacc} ")
            return train_losses, train_accuracies, val_losses, val_accuracies, (epoch+1)

    

            


