from data_acquisition import load_breastmnist, load_dataset_CNN, load_dataset_ml
from training import trainCNN, kernel_svm
from plotting import plot_loss_acc, class_balance
from sklearn.metrics import classification_report
import numpy as np
seed=[0,1,2,3,4]
batch_size=[32,64,128]
result = []



f1_svm_seeds = []
recall_svm_seeds = []
accuracy_svm_seeds = []


for batch in batch_size:
    train_ds, val_ds, test_ds = load_breastmnist(batch)
    train, val,test = load_dataset_CNN(train_ds, val_ds, test_ds, batch)
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_ml(train_ds, val_ds, test_ds)
    class_balance(train.dataset.labels)
    for s in seed:


        train_loss, train_acc, val_loss, val_acc, epoch, best_prediction = trainCNN(0.0001, 100, train, val, 1, 2, s)

        result.append({
            "seed":s,
            "train_loss":train_loss,
            "train_acc" : train_acc,
            "val_loss" : val_loss,
            "val_acc" : val_acc
        })
        print(classification_report(val.dataset.labels, best_prediction))


        f1, accuracy, recall = kernel_svm(x_train, y_train, x_val, y_val, s)
        f1_svm_seeds.append(f1)
        recall_svm_seeds.append(recall)
        accuracy_svm_seeds.append(accuracy)
    
    mean_svm_accuracy = np.mean(accuracy_svm_seeds)
    mean_svm_f1 = np.mean(f1_svm_seeds)
    mean_svm_recall = np.mean(recall_svm_seeds)
    print(f"accuracy: {mean_svm_accuracy}     f1:{mean_svm_f1}       recall:{mean_svm_recall}")
    plot_loss_acc(result, f'initial{batch}')
    result.clear()


