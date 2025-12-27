import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_loss_acc(result, filename:str):

    """
    Plot loss and accuracy for CNN

    arguments:
    result: list with accuracy and loss for training and validation of each seed
    filename: unique filename used for this run to save plots

    """

    train_loss = [r['train_loss'] for r in result]
    val_loss = [r['val_loss'] for r in result]
    train_acc = [r['train_acc'] for r in result]
    val_acc = [r['val_acc']for r in result]
    min_len = min(len(l) for l in train_loss)
    train_loss = [l[:min_len] for l in train_loss]
    val_loss   = [l[:min_len] for l in val_loss]

    train_acc = [l[:min_len] for l in train_acc]
    val_acc   = [l[:min_len] for l in val_acc]
    train_loss_arr = np.array(train_loss)
    val_loss_arr = np.array(val_loss)

    train_acc_arr = np.array(train_acc)
    val_acc_arr = np.array(val_acc)

    train_loss_mean = train_loss_arr.mean(axis=0)
    val_loss_mean = val_loss_arr.mean(axis=0)
    train_acc_mean = train_acc_arr.mean(axis=0)
    val_acc_mean = val_acc_arr.mean(axis=0)

    plt.figure(figsize=(12, 5)) 
    for i in range(train_loss_arr.shape[0]):
        plt.plot(range(min_len),train_loss_arr[i], color='blue', alpha=0.2)
        plt.plot(range(min_len), val_loss_arr[i], color='orange', alpha=0.2)
    plt.plot(range(min_len), train_loss_mean, color='blue', linewidth=2, label='train loss mean')
    plt.plot(range(min_len), val_loss_mean, color='orange', linewidth=2, label='validation loss mean')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_val_loss_{filename}.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    for i in range(train_loss_arr.shape[0]):
        plt.plot(range(min_len),train_acc_arr[i], color='blue', alpha=0.3)
        plt.plot(range(min_len), val_acc_arr[i], color='orange',  alpha=0.3)

    plt.plot(range(min_len), train_acc_mean, color='blue', linewidth=2, label='train acc mean')
    plt.plot(range(min_len), val_acc_mean, color='orange', linewidth=2, label='validation acc mean')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_val_acc_{filename}.png")
    plt.close()

def class_balance(labels):

    y = pd.Series(labels.reshape(-1))
    freqs = y.value_counts().sort_index() / len(y)

    plt.bar(freqs.index.astype(str), freqs.values)
    plt.ylabel("Proportion")
    plt.title("Class balance")
    plt.savefig("class_balance.png")

    print("Counts:\n", y.value_counts().sort_index())
    print("Proportions:\n", freqs)
