import medmnist
import numpy as np
from medmnist.dataset import BreastMNIST
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 


def load_breastmnist(batch_size):
    transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize(mean=[0.5], std=[0.5])])
    
    train_ds = BreastMNIST(split="train", transform=transform, download=True)
    val_ds   = BreastMNIST(split="val",   transform=transform, download=True)
    test_ds  = BreastMNIST(split="test",  transform=transform, download=True)

    return train_ds, val_ds, test_ds


def load_dataset_CNN(train_ds, val_ds, test_ds, batch_size):
        # data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_dataset_ml(train,val,test):
    x_train = train.imgs.astype(np.float32)
    y_train=train.labels.squeeze().astype(np.int64)

    x_val = val.imgs.astype(np.float32)
    y_val = val.labels.squeeze().astype(np.int64)

    x_test = test.imgs.astype(np.float32)
    y_test = test.labels.squeeze().astype(np.int64)

    x_train = x_train.reshape(x_train.shape[0], -1)  # (546, 784)
    x_val = x_val.reshape(x_val.shape[0], -1)        # (78, 784)
    x_test = x_test.reshape(x_test.shape[0], -1)     # (156, 784)

    x_train /= 255.0
    x_val   /= 255.0
    x_test  /= 255.0

    x_train = (x_train - 0.5) / 0.5
    x_val   = (x_val   - 0.5) / 0.5
    x_test  = (x_test  - 0.5) / 0.5

    return x_train, y_train, x_val, y_val, x_test, y_test

def augment_data(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize(mean=[0.5], std=[0.5])])
    train_ds = BreastMNIST(split="train", transform=train_transform, download=True)
    val_ds   = BreastMNIST(split="val",   transform=val_transform, download=True)
    test_ds  = BreastMNIST(split="test",  transform=val_transform, download=True)

    return train_ds, val_ds, test_ds




    

            
