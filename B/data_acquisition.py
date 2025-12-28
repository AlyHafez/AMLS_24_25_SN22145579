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


def augment_data(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
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




    

            
