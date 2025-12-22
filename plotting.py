import matplotlib.pyplot as plt
import numpy as np


def plot_train_loss(train_loss, train_acc, val_loss, val_acc, epoch, filename:str):
    plt.figure(figsize=(12, 5))
    plt.plot(range(epoch),np.array(train_loss), color='blue', label='train loss')
    plt.plot(range(epoch), np.array(val_loss), color='orange', label='validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_val_loss_{filename}.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(range(epoch),np.array(train_acc), color='blue', label='train accuracy')
    plt.plot(range(epoch), np.array(val_acc), color='orange', label='validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_val_acc_{filename}.png")
    plt.close()

