# !/use/bin/env python
# encoding:utf-8
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import *
from sklearn.externals import joblib
from sklearn import metrics

from sklearn.model_selection import GridSearchCV


def SVM_classifier(X_train, y_train, cross_validation_value, CPU_value):
    
    svc = svm.SVC(probability=True)
    #Set the parameters of the SVM
    parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-5,15,11)), 'gamma':map(lambda x:2**x,np.linspace(3,-15,7))}
    clf = GridSearchCV(svc, parameters, cv=cross_validation_value, n_jobs=CPU_value, scoring='accuracy')
    clf.fit(X_train, y_train)
    #save model
    joblib.dump(clf,"SVM_best.model")
    C=clf.best_params_['C']
    gamma=clf.best_params_['gamma']
    print('c:',C,'gamma:',gamma)
    y_predict=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,),X_train,y_train,cv=cross_validation_value,n_jobs=CPU_value)
    y_predict_prob=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X_train,y_train,cv=cross_validation_value,n_jobs=CPU_value,method='predict_proba')

    ACC=metrics.accuracy_score(y_train,y_predict)
    print(ACC)
    
if __name__=="__main__":
    #Some parameters
    inputfile_path="./enhancer_sup_wN8_cv.csv"
    crossvalidation_values=5
    CPU_values=8
    #read dataset
    train_data = pd.read_csv(inputfile_path, header=None, index_col=None)
    
    X =train_data.iloc[:,1:]
    Y= train_data.iloc[:,[0]]
    X = X.values
    Y=Y.values
    Y =Y.reshape(-1)
    #SVM
    SVM_classifier(X,Y, crossvalidation_values, CPU_values)
    


