import medmnist
import numpy as np
from medmnist import BreastMNIST
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 

def load_breastmnist(batch_size=64):
    transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize(mean=[0.5], std=[0.5])])
    
    train_ds = BreastMNIST(split="train", transform=transform, download=True)
    val_ds   = BreastMNIST(split="val",   transform=transform, download=True)
    test_ds  = BreastMNIST(split="test",  transform=transform, download=True)

    # data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def normalize_data(train,val, test):
    transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize(mean=[0.5], std=[0.5])])
    
    