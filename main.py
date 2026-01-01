from A.data_acquisition_a import load_breastmnist_ml, load_dataset_ml, PCA_ml
from A.train_val_svm import kernel_svm, performance, test_svm
from B.data_acquisition import load_breastmnist, load_dataset_CNN, augment_data
from B.training import trainCNN, evaluate_CNN
from B.plotting import plot_loss_acc, class_balance, statistics, test_performance
from sklearn.metrics import classification_report
import numpy as np
seed=[0,1,2,3,4]

test_acc = []
test_f1 = []
test_recall = []
test_prec = []
test_auc = []
model_medium = []
model_large = []
model_med_aug = []
model_large_aug = []


small_cnn = [8,16,32]
medium_cnn = [16,32,64]
large_cnn = [32,64,128]
very_large_cnn = [64,128,256]

result_small = []
result_medium = []
result_large = []
result_very_large = []
train_ds, val_ds, test_ds = load_breastmnist()

    


train, val,test = load_dataset_CNN(train_ds, val_ds, test_ds, batch_size=64)  
for s in seed:
    train_loss, train_acc, val_loss, val_acc, epoch, best_prediction, bestacc, model,auc = trainCNN(0.0001, 150, train, val, 1, 2,small_cnn, s)
    result_small.append({
    "seed":s,
    "train_loss":train_loss,
    "train_acc" : train_acc,
    "val_loss" : val_loss,
    "val_acc" : val_acc,
    "best_prediction" : best_prediction,
    "bestacc" : bestacc,
    "auc": auc
    })

    train_loss, train_acc, val_loss, val_acc, epoch, best_prediction, bestacc, model, auc = trainCNN(0.0001, 150, train, val, 1, 2,medium_cnn, s)
    
    model_medium.append(model)
    result_medium.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc,
        "best_prediction" : best_prediction,
        "bestacc" : bestacc,
        "auc": auc
    })

    train_loss, train_acc, val_loss, val_acc, epoch, best_prediction, bestacc, model,auc = trainCNN(0.0001, 150, train, val, 1, 2,large_cnn, s)
    model_large.append(model)
    result_large.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc,
        "best_prediction" : best_prediction,
        "bestacc" : bestacc,
        "auc": auc
    })

    train_loss, train_acc, val_loss, val_acc, epoch, best_prediction, bestacc, model,auc = trainCNN(0.0001, 150, train, val, 1, 2,very_large_cnn, s)
    
    result_very_large.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc,
        "best_prediction" : best_prediction,
        "bestacc" : bestacc,
        "auc": auc
    })

statistics(val, result_small, 'small_cnn')
plot_loss_acc(result_small, 'small_cnn')
result_small.clear()
statistics(val, result_medium, 'medium_cnn')
plot_loss_acc(result_medium, 'medium_cnn')
result_medium.clear()
statistics(val, result_large, 'large_cnn')
plot_loss_acc(result_large, 'large_cnn')
result_large.clear()
statistics(val, result_very_large, 'very_large_cnn')
plot_loss_acc(result_very_large, 'very_large_cnn')
result_very_large.clear()




train_ml, val_ml, test_ml = load_breastmnist_ml()
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_ml(train_ml, val_ml, test_ml)
y_pred_lin, y_score_lin, svm_model_lin = kernel_svm(x_train, y_train, x_val, seed=0, kernel="linear")
y_pred, y_score, svm_model = kernel_svm(x_train, y_train, x_val, seed=0, kernel="rbf")
print("Linear Kernel SVM")
performance(y_val, y_pred_lin, y_score_lin)
print("RBF Kernel SVM")
performance(y_val, y_pred, y_score)



aug_train_ds, aug_val_ds, aug_test_ds = augment_data()
train_aug, val_aug,test_aug = load_dataset_CNN(aug_train_ds, aug_val_ds, aug_test_ds, batch_size=64)
train_ml, val_ml, test_ml = load_breastmnist_ml()
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_ml(train_ml, val_ml, test_ml)




for s in seed:
   
    train_loss, train_acc, val_loss, val_acc, best_epoch, best_prediction, bestacc, model_augment,auc = trainCNN(0.0001, 150, train_aug, val_aug, 1, 2,medium_cnn, s)
    model_med_aug.append(model_augment)
    result_medium.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc,
        "best_prediction" : best_prediction,
        "bestacc" : bestacc,
        "auc": auc
    })
    train_loss, train_acc, val_loss, val_acc, best_epoch, best_prediction, bestacc, model_augment,auc = trainCNN(0.0001, 150, train_aug, val_aug, 1, 2,large_cnn, s)
    model_large_aug.append(model_augment)

    result_large.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc,
        "best_prediction" : best_prediction,
        "bestacc" : bestacc,
        "auc": auc
    })
    print(classification_report(val_aug.dataset.labels, best_prediction))

x_train_pca, x_val_pca, x_test_pca = PCA_ml(x_train, x_val, x_test, seed=0)
y_pred, y_score, svm_model_pca_lin = kernel_svm(x_train_pca, y_train, x_val_pca, seed=0, kernel="linear")
performance(y_val, y_pred, y_score)
y_pred,y_score, svm_model_pca = kernel_svm(x_train_pca, y_train, x_val_pca, seed=0, kernel="rbf")
performance(y_val, y_pred, y_score)
statistics(val_aug, result_medium, 'augmented medium')

plot_loss_acc(result_medium, 'augmented medium')

statistics(val_aug, result_large, 'augmented large')

plot_loss_acc(result_large, 'augmented large')
result_medium.clear()
result_large.clear()



print("Final test evaluation:")

y_test_pred_lin, y_score = test_svm(x_test,svm_model_lin)
    
print("Linear SVM without PCA")
performance(y_test, y_test_pred_lin, y_score)

print("RBF SVM without PCA")  
y_test_pred_rbf, y_score = test_svm(x_test,svm_model)


performance(y_test, y_test_pred_rbf, y_score)

print("linear SVM with PCA")
y_test_pred_lin, y_score = test_svm(x_test_pca,svm_model_pca_lin)
performance(y_test, y_test_pred_lin, y_score)
print("RBF SVM with PCA")
y_test_pred, y_score = test_svm(x_test_pca,svm_model_pca)
performance(y_test, y_test_pred, y_score)




train, val, test = load_dataset_CNN(train_ds, val_ds, test_ds, batch_size=64)



for m in model_medium:
   
    acc, prec, rec, f1, cm, auc = evaluate_CNN(m, test)
    test_acc.append(acc)
    test_prec.append(prec)
    test_recall.append(rec)
    test_f1.append(f1)
    test_auc.append(auc)

print("Medium / No augmentation: ")
test_performance(test_acc,test_prec,test_recall,test_f1, test_auc)
test_acc.clear()
test_prec.clear()
test_recall.clear()
test_f1.clear()
test_auc.clear()

for m in  model_large:
    
    acc, prec, rec, f1, cm, auc = evaluate_CNN(m, test)
    test_acc.append(acc)
    test_prec.append(prec)
    test_recall.append(rec)
    test_f1.append(f1)
    test_auc.append(auc)

print("Large / No augmentation: ")
test_performance(test_acc,test_prec,test_recall,test_f1, test_auc)
test_acc.clear()
test_prec.clear()
test_recall.clear()
test_f1.clear()
test_auc.clear()


for m in  model_med_aug:
    
    acc, prec, rec, f1, cm, auc = evaluate_CNN(m, test)
    test_acc.append(acc)
    test_prec.append(prec)
    test_recall.append(rec)
    test_f1.append(f1)
    test_auc.append(auc)

print("Medium / Augmentation: ")
test_performance(test_acc,test_prec,test_recall,test_f1, test_auc)
test_acc.clear()
test_prec.clear()
test_recall.clear()
test_f1.clear()
test_auc.clear()
for m in  model_large_aug:
    
    acc, prec, rec, f1, cm, auc = evaluate_CNN(m, test)
    test_acc.append(acc)
    test_prec.append(prec)
    test_recall.append(rec)
    test_f1.append(f1)
    test_auc.append(auc)

print("Large / Augmentation: ")
test_performance(test_acc,test_prec,test_recall,test_f1, test_auc)
test_acc.clear()
test_prec.clear()
test_recall.clear()
test_f1.clear()
test_auc.clear()










    

