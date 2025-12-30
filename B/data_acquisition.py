import medmnist
import numpy as np
from medmnist.dataset import BreastMNIST
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 


def load_breastmnist():
    """
    Download dataset and apply preprocessing transforms

    Images are first converted to tensors with values in [0, 1] using ToTensor(),
    then normalized to the range [-1, 1] using mean=0.5 and std=0.5.

    returns:
    train_ds : training dataset
    val_ds: validation dataset 
    test_ds: test dataset    
    """

    transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize(mean=[0.5], std=[0.5])])
    
    train_ds = BreastMNIST(split="train", transform=transform, download=True)
    val_ds   = BreastMNIST(split="val",   transform=transform, download=True)
    test_ds  = BreastMNIST(split="test",  transform=transform, download=True)

    return train_ds, val_ds, test_ds


def load_dataset_CNN(train_ds, val_ds, test_ds, batch_size):
    """
    Create PyTorch DataLoader for train, val and test datasets

    The DataLoader handles batching and shuffling of data during training and evaluation.
    training data is shuffled to improve model generalization.
    while validation and test data are not shuffled to maintain consistent evaluation.
    parameters:
    train_ds : training dataset
    val_ds: validation dataset
    test_ds: test dataset
    batch_size: size of each data batch
    returns:
    train_loader : DataLoader for training dataset
    val_loader: DataLoader for validation dataset
    test_loader: DataLoader for test dataset
    
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def augment_data():
    """
    Apply data augmentation techniques to the training dataset to enhance model generalization.
    returns:
    train_ds : augmented training dataset
    val_ds: validation dataset
    test_ds: test dataset

    """
    
    train_transform = transforms.Compose([
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5)],
            p=0.5
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.11, contrast=0.11)],
            p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])



    val_transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize(mean=[0.5], std=[0.5])])
    train_ds = BreastMNIST(split="train", transform=train_transform, download=True)
    val_ds   = BreastMNIST(split="val",   transform=val_transform, download=True)
    test_ds  = BreastMNIST(split="test",  transform=val_transform, download=True)

    return train_ds, val_ds, test_ds




    

            
