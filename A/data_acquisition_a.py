import medmnist
import numpy as np
from medmnist.dataset import BreastMNIST
from sklearn.decomposition import PCA



def load_breastmnist_ml():

    """
    loads train, val, test datasets as raw images from MEDMNIST
    
    Returns:
    train, val, test datasets
     """
    train_ds = BreastMNIST(split="train",  download=True)
    val_ds   = BreastMNIST(split="val",   download=True)
    test_ds  = BreastMNIST(split="test",  download=True)

    return train_ds, val_ds, test_ds

def load_dataset_ml(train,val,test):

    """
    Converts raw images in datasets into np arrays suitable for use for the Machine learning model and splits features from labels

    parameters: 
        train dataset, val dataset, test dataset

    Returns:
    x_train,y_train, x_val y_val, x_test, y_test 
     """
    # convert raw image to a set of floats with dimensions (n, Height, Width), and labels as a integer
    x_train = train.imgs.astype(np.float32) 
    y_train=train.labels.squeeze().astype(np.int64)

    x_val = val.imgs.astype(np.float32)
    y_val = val.labels.squeeze().astype(np.int64)

    x_test = test.imgs.astype(np.float32)
    y_test = test.labels.squeeze().astype(np.int64)

    # reshape array from (n,  Height, Width) to (n, Height* Width) making it suitable to input into ML model
    x_train = x_train.reshape(x_train.shape[0], -1)  # (546, 784) 
    x_val = x_val.reshape(x_val.shape[0], -1)        # (78, 784)
    x_test = x_test.reshape(x_test.shape[0], -1)     # (156, 784)

    # normalize datapoints so that they vary from [-1,1] instead of [0,255]
    x_train /= 255.0 
    x_val   /= 255.0
    x_test  /= 255.0
    x_train = (x_train - 0.5) / 0.5
    x_val   = (x_val   - 0.5) / 0.5
    x_test  = (x_test  - 0.5) / 0.5

    return x_train, y_train, x_val, y_val, x_test, y_test

def PCA_ml(x_train, x_val, x_test, seed):
    """
    Perform PCA on dataset to reduce dimensionality for ml

    PCA is fitted only on the training data to avoid information leakage.
    Validation and test sets are transformed using the same PCA model.

    parameters: 
    x_train (ndarray): Training feature matrix
    x_val (ndarray): Validation feature matrix
    x_test (ndarray): Test feature matrix
    seed (int): Random seed for reproducibility

    Returns:
    x_train_pca (ndarray): PCA-transformed training features
    x_val_pca (ndarray): PCA-transformed validation features
    x_test_pca (ndarray): PCA-transformed test features
     """   
    
    pca = PCA(n_components=0.95, random_state=seed) # perform PCA until sum of all variance ratios reach 95%
    x_train_pca = pca.fit_transform(x_train)# fit PCA using training set and transform training set
    cumulative = np.cumsum(pca.explained_variance_ratio_) # calculate cummulative sum of the variance ratio for analysis
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)
    for i, var in enumerate(cumulative):
        print(f"PC1–PC{i+1}: {var:.4f} ({var*100:.2f}%)") # print number of features after PCA with the cummulative sum of the variance ratio
    return x_train_pca, x_val_pca, x_test_pca