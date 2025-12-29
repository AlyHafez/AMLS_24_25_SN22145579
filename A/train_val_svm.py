from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, recall_score, precision_score
from sklearn import svm
import numpy as np

def kernel_svm(x_train, y_train, x_val, seed):
    """
    train and validate kernel SVM using training and validation
    returns both the prediction on validation set and the trained model for running test set further on


    parameters: 
    x_train (ndarray): Training feature matrix
    y_train (ndarray): training labels to train model
    x_val (ndarray): Validation feature matrix
    
    seed (int): Random seed for reproducibility

    Returns:
    y_pred (ndarray): predicted validation class labels for validation set
    clf : Trained SVM classifier
     """   
    clf = svm.SVC(kernel="rbf",C=1.0,gamma="scale",class_weight="balanced",random_state=seed)      
    clf.fit(x_train, y_train)   
    y_pred = clf.predict(x_val)

    return y_pred, clf

def test_svm(x_test, model):
    """
    run previously saved trained SVM on test set for final performance


    parameters: 
    x_test (ndarray): Test feature set
    model: Trained SVM classifier

    Returns:
    y_pred (ndarray): predicted  class labels for test set
    
     """   
    y_pred = model.predict(x_test)
    return y_pred

def performance(y_true, y_pred):
    """
    calculate performance metrics for predictions made by SVM 
    It then prints put performance metrics such as accuracy, precision, recall, f1 and the confusion matrix for performance analysis


    parameters: 
    y_true (ndarray): True class labels for given set
    y_pred (ndarray): Predicted class labels for given set made by SVM classifier

     """   
        
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
