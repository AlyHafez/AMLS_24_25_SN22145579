
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
model_orig = []
model_aug = []




for batch in batch_size:
    train_ds, val_ds, test_ds = load_breastmnist(batch)
    
    train, val,test = load_dataset_CNN(train_ds, val_ds, test_ds, batch)
    
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
        if batch ==64:
            model_orig.append(model)

        print(classification_report(val.dataset.labels, best_prediction))


    
        
    

    statistics(val, result, 'initial')
    plot_loss_acc(result, f'initial{batch}')
    result.clear()
train_ml, val_ml, test_ml = load_breastmnist_ml()
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_ml(train_ml, val_ml, test_ml)
y_pred, svm_model = kernel_svm(x_train, y_train, x_val, y_val, seed=0)
performance(y_val, y_pred)



aug_train_ds, aug_val_ds, aug_test_ds = augment_data(batch_size=64)
train_aug, val_aug,test_aug = load_dataset_CNN(aug_train_ds, aug_val_ds, aug_test_ds, batch_size=64)
train_ml, val_ml, test_ml = load_breastmnist_ml()
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_ml(train_ml, val_ml, test_ml)




for s in seed:
   
    train_loss, train_acc, val_loss, val_acc, best_epoch, best_prediction, bestacc, model_augment = trainCNN(0.0001, 100, train_aug, val_aug, 1, 2, s)
    model_aug.append(model_augment)
    result.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc,
        "best_prediction" : best_prediction,
        "bestacc" : bestacc
    })
    print(classification_report(val_aug.dataset.labels, best_prediction))
x_train_pca, x_val_pca, x_test_pca = PCA_ml(x_train, x_val, x_test, seed=0)
y_pred, svm_model_pca = kernel_svm(x_train_pca, y_train, x_val_pca, y_val, seed=0)
    
statistics(val_aug, result, 'augmented')
performance(y_val, y_pred)
plot_loss_acc(result, 'augmented')
result.clear()



print("Final test evaluation:")

y_test_pred = test_svm(x_test,svm_model)
    
print("SVM without PCA")
performance(y_test, y_test_pred)

  
y_test_pca = test_svm(x_test_pca,svm_model_pca)

print("SVM with PCA")
performance(y_test, y_test_pca)


train_ds, val_ds, test_ds = load_breastmnist(batch_size=64)
train, val, test = load_dataset_CNN(train_ds, val_ds, test_ds, batch_size=64)

aug_train_ds, aug_val_ds, aug_test_ds = augment_data(batch_size=64)
aug_train, aug_val, aug_test = load_dataset_CNN(aug_train_ds, aug_val_ds, aug_test_ds, batch_size=64)
for m in model_orig:
   
    acc, prec, rec, f1, cm = evaluate_CNN(m, test)
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

for m in  model_aug:
    
    acc, prec, rec, f1, cm = evaluate_CNN(m, test)
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











    

