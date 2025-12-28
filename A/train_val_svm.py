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

def performance(y_true, y_pred):

        
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true,y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)


    print("SVM performance:")
    print(f"accuracy SVM is: {accuracy}")
    print(f"precision SVM is : {precision}")
    print(f"Recall SVM is: {recall}")
    print(f"F1 SVM is: {f1}")
    print(confusion_matrix(y_true, y_pred))
