
from A.data_acquisition_a import load_breastmnist_ml, load_dataset_ml, PCA_ml
from A.train_val_svm import kernel_svm, performance, test_svm
from B.data_acquisition import load_breastmnist, load_dataset_CNN, augment_data
from B.training import trainCNN, evaluate_CNN
from B.plotting import plot_loss_acc, class_balance, statistics, test_performance
from sklearn.metrics import classification_report
import numpy as np
seed=[0,1,2,3,4]
batch_size=[32,64,128]
result = []
test_acc = []
test_f1 = []
test_recall = []
test_prec = []


svm_pred = []


for batch in batch_size:
    train_ds, val_ds, test_ds = load_breastmnist(batch)
    train_ml, val_ml, test_ml = load_breastmnist_ml()
    train, val,test = load_dataset_CNN(train_ds, val_ds, test_ds, batch)
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_ml(train_ml, val_ml, test_ml)
    class_balance(train.dataset.labels, 'initial')
    for s in seed:


        train_loss, train_acc, val_loss, val_acc, epoch, best_prediction, bestacc, model = trainCNN(0.0001, 100, train, val, 1, 2, s)

        result.append({
            "seed":s,
            "train_loss":train_loss,
            "train_acc" : train_acc,
            "val_loss" : val_loss,
            "val_acc" : val_acc,
            "best_prediction" : best_prediction,
            "bestacc" : bestacc
        })
        print(classification_report(val.dataset.labels, best_prediction))


        y_pred, svm_model = kernel_svm(x_train, y_train, x_val, y_val, s)
        svm_pred.append(y_pred)
    

    statistics(val, result, 'initial')
    plot_loss_acc(result, f'initial{batch}')
    performance(y_val, svm_pred)
    result.clear()
    svm_pred.clear()


aug_train_ds, aug_val_ds, aug_test_ds = augment_data(batch_size=64)
train_aug, val_aug,test_aug = load_dataset_CNN(aug_train_ds, aug_val_ds, aug_test_ds, batch_size=64)
train_ml, val_ml, test_ml = load_breastmnist_ml()
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_ml(train_ml, val_ml, test_ml)


class_balance(train.dataset.labels, 'after_aug')

for s in seed:
    x_train_pca, x_val_pca, x_test_pca = PCA_ml(x_train, x_val, x_test, s)
    train_loss, train_acc, val_loss, val_acc, best_epoch, best_prediction, bestacc, model = trainCNN(0.0001, 100, train_aug, val_aug, 1, 2, s)
    result.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc,
        "best_prediction" : best_prediction,
        "bestacc" : bestacc
    })
    print(classification_report(val.dataset.labels, best_prediction))
    y_pred, svm_model_pca = kernel_svm(x_train_pca, y_train, x_val_pca, y_val, s)
    svm_pred.append(y_pred)
statistics(val_aug, result, 'augmented')
performance(y_val, svm_pred)
plot_loss_acc(result, 'augmented')
result.clear()
svm_pred.clear()


print("Final test evaluation:")
for s in seed:    
    y_test_pred = test_svm(x_test,svm_model)
    svm_pred.append(y_test_pred)
print("SVM without PCA")
performance(y_test, svm_pred)
svm_pred.clear()
for s in seed:    
    y_test_pca = test_svm(x_test_pca,svm_model_pca)
    svm_pred.append(y_test_pca)
print("SVM with PCA")
performance(y_test, svm_pred)
svm_pred.clear()

train_ds, val_ds, test_ds = load_breastmnist(batch_size=64)
train, val, test = load_dataset_CNN(train_ds, val_ds, test_ds, batch_size=64)

aug_train_ds, aug_val_ds, aug_test_ds = augment_data(batch_size=64)
aug_train, aug_val, aug_test = load_dataset_CNN(aug_train_ds, aug_val_ds, aug_test_ds, batch_size=64)
for s in seed:
    _, _, _, _, _, _, _, best_model = trainCNN(0.0001, 100, train, val, 1, 2, s)
    acc, prec, rec, f1, cm = evaluate_CNN(best_model, test, s)
    test_acc.append(acc)
    test_prec.append(prec)
    test_recall.append(rec)
    test_f1.append(f1)

print("No augmentation: ")
test_performance(test_acc,test_prec,test_recall,test_f1)
test_acc.clear()
test_prec.clear()
test_recall.clear()
test_f1.clear()

for s in seed:
    _, _, _, _, _, _, _, best_model_aug = trainCNN(0.0001, 100, aug_train, aug_val, 1, 2, s)
    acc, prec, rec, f1, cm = evaluate_CNN(best_model_aug, test, s)
    test_acc.append(acc)
    test_prec.append(prec)
    test_recall.append(rec)
    test_f1.append(f1)

print("augmented: ")
test_performance(test_acc,test_prec,test_recall,test_f1)
test_acc.clear()
test_prec.clear()
test_recall.clear()
test_f1.clear()











    

