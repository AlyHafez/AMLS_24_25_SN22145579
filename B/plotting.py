import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, recall_score, precision_score

def plot_loss_acc(result, filename:str):

    """
    Plot loss and accuracy for CNN

    arguments:
    result: list with accuracy and loss for training and validation of each seed
    filename: unique filename used for this run to save plots under unqiue names

    """

    train_loss = [r['train_loss'] for r in result] # extract train loss from each seed's result
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

    train_acc_arr = np.array(train_acc) # convert list of lists to numpy array for easier manipulation
    val_acc_arr = np.array(val_acc)

    train_loss_mean = train_loss_arr.mean(axis=0) # compute mean across seeds for each epoch
    val_loss_mean = val_loss_arr.mean(axis=0)
    train_acc_mean = train_acc_arr.mean(axis=0)
    val_acc_mean = val_acc_arr.mean(axis=0)

    plt.figure(figsize=(12, 5)) 
    for i in range(train_loss_arr.shape[0]):# plot individual seed runs with low opacity
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

    plt.figure(figsize=(12, 5))# plot accuracy
    for i in range(train_loss_arr.shape[0]):
        plt.plot(range(min_len),train_acc_arr[i], color='blue', alpha=0.3)
        plt.plot(range(min_len), val_acc_arr[i], color='orange',  alpha=0.3)

    plt.plot(range(min_len), train_acc_mean, color='blue', linewidth=2, label='train acc mean') # plot mean accuracy across seeds
    plt.plot(range(min_len), val_acc_mean, color='orange', linewidth=2, label='validation acc mean')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_val_acc_{filename}.png")
    plt.close()

def class_balance(labels, filename:str):
    """
    Plot class balance bar chart and print counts and proportions
    
    parameters:
    labels (ndarray): class labels for given dataset
    filename (str): unique filename used for this run to save plots under unqiue names

    """
    y = pd.Series(labels.reshape(-1)) # convert to pandas series for easier manipulation
    freqs = y.value_counts().sort_index() / len(y) # calculate proportion of each class

    plt.bar(freqs.index.astype(str), freqs.values)# plot bar chart
    plt.ylabel("Proportion")
    plt.title("Class balance")
    plt.savefig(f"class_balance{filename}.png")

    print("Counts:\n", y.value_counts().sort_index())
    print("Proportions:\n", freqs)

def statistics(ref, result, filename:str):
    """
    Calculate and print performance metrics for CNN across different seeds such as accuracy, precision, recall, f1 and their mean and std across different seeds
    
    parameters:
    ref: reference dataset with true labels
    result: list with accuracy and loss for training and validation of each seed
    filename: unique filename used for this run to save plots under unqiue names

    """
    y_true = ref.dataset.labels.squeeze() # extract true labels from reference dataset


    prec_list = []
    rec_list = []
    f1_list = []
    best_acc = [r['bestacc'] for r in result] # extract best accuracy from each seed's result
    
    mean_acc = np.mean(best_acc) # compute mean and std accuracy across seeds
    acc_std = np.std(best_acc)
    # compute precision, recall, f1 for each seed and store in lists
    for r in result:
        y_pred = np.array(r["best_prediction"]).squeeze() # extract predicted labels from each seed's result


        prec_list.append(precision_score(y_true, y_pred, average="macro", zero_division=0)) 
        rec_list.append(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_list.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
    # print classification report for each seed
    print(f"CNN performance for {filename}")
    print(f"Accuracy:  {mean_acc} ± {acc_std}")
    print(f"Precision: {np.mean(prec_list):.3f} ± {np.std(prec_list):.3f}")
    print(f"Recall:    {np.mean(rec_list):.3f} ± {np.std(rec_list):.3f}")
    print(f"Macro F1:  {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")


   
def test_performance(acc_list:list, prec_list:list, rec_list:list,f1_list:list):
    """
    calculate and print performance metrics mean and std for CNN across different seeds such as accuracy, precision, recall, f1 for test set
    
    parameters:
    acc_list (list): list of accuracy values across different seeds
    prec_list (list): list of precision values across different seeds
    rec_list (list): list of recall values across different seeds
    f1_list (list): list of f1 values across different seeds

    """
    # print performance metrics with mean and std across different seeds
    print(f"Accuracy: {np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}")
    print(f"Precision: {np.mean(prec_list):.3f} ± {np.std(prec_list):.3f}")
    print(f"Recall:    {np.mean(rec_list):.3f} ± {np.std(rec_list):.3f}")
    print(f"Macro F1:  {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")
