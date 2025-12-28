from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, recall_score, precision_score
from sklearn import svm
import numpy as np

def kernel_svm(x_train, y_train, x_val,y_val, seed):
    clf = svm.SVC(kernel="rbf",C=1.0,gamma="scale",class_weight="balanced",random_state=seed)      
    clf.fit(x_train, y_train)   
    y_pred = clf.predict(x_val)

    return y_pred, clf

def test_svm(x_test, model):
    y_pred = model.predict(x_test)
    return y_pred

def performance(y_val, y_pred):
    f1_seed =[]
    recall_seed = []
    precision_seed = []
    accuracy_seed =[]
    for y in y_pred:
        
        f1 = f1_score(y_val, y, average='macro')
        accuracy = accuracy_score(y_val, y)
        recall = recall_score(y_val,y, average="macro")
        precision = precision_score(y_val, y, average="macro", zero_division=0)
        f1_seed.append(f1)
        accuracy_seed.append(accuracy)
        recall_seed.append(recall)
        precision_seed.append(precision)
    
    mean_f1 = np.mean(f1_seed)
    std_f1 = np.std(f1_seed)

    mean_recall = np.mean(recall_seed)
    std_recall = np.std(recall_seed)

    mean_acc_svm = np.mean(accuracy_seed)
    std_acc_svm = np.std(accuracy_seed)

    mean_prec_svm= np.mean(precision_seed)
    std_prec_svm = np.std(precision_seed)
    print("SVM performance:")
    print(f"accuracy SVM is: {mean_acc_svm} ± {std_acc_svm}")
    print(f"precision SVM is : {mean_prec_svm}±{std_prec_svm}")
    print(f"Recall SVM is: {mean_recall} ± {std_recall}")
    print(f"F1 SVM is: {mean_f1} ± {std_f1}")
    print(confusion_matrix(y_val, y))
