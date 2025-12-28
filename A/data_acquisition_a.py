import medmnist
import numpy as np
from medmnist.dataset import BreastMNIST
from sklearn.decomposition import PCA



def load_breastmnist_ml():

    
    train_ds = BreastMNIST(split="train",  download=True)
    val_ds   = BreastMNIST(split="val",   download=True)
    test_ds  = BreastMNIST(split="test",  download=True)

    return train_ds, val_ds, test_ds

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

def PCA_ml(x_train, x_val, x_test, seed):
    pca = PCA(n_components=0.95, random_state=seed)
    x_train_pca = pca.fit_transform(x_train)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)
    for i, var in enumerate(cumulative):
        print(f"PC1–PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    return x_train_pca, x_val_pca, x_test_pca