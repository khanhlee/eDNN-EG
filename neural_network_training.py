# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:43:01 2019

@author: TMU
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Dropout, Flatten


trn_file = 'essential.wN6.cv.csv'

# Define parameters
num_epochs = 50
nb_classes = 2
nb_kernels = 3
nb_pools = 2
num_features = 100

def load_data(trn):
    train = pd.read_csv(trn, header=None)
    X_trn = train.iloc[:,1:num_features+1]
    Y_trn = train.iloc[:,0]
    
    return X_trn, Y_trn

def libsvm_model():
    clf = SVC(C=23768, gamma=0.001953125, kernel='rbf', probability=True)
    return clf

def knn_model():
    knn = KNeighborsClassifier(n_neighbors=10)
    return knn

def randomforest_model():
    rf = RandomForestClassifier(bootstrap=True, max_features='auto', n_estimators=500)
    return rf

def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, init='uniform', activation='relu', input_dim=400))
    model.add(Dense(100, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, init='uniform', activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_model():
    model = Sequential()

    model.add(Conv1D(32, 3, activation='relu', input_shape=(num_features,1)))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

X_trn, y_trn = load_data(trn_file)
kfold = StratifiedKFold(n_splits=5, shuffle=True)


# Traditional machine learning training
for train, test in kfold.split(X_trn, y_trn):
#    train_x, train_y = X_trn.iloc[train], Y_trn.iloc[train]
    svm_model = libsvm_model()   
    ## evaluate the model
    svm_model.fit(X_trn.iloc[train], y_trn.iloc[train])
    # evaluate the model
    true_labels = np.asarray(y_trn.iloc[test])
    predictions = svm_model.predict(X_trn.iloc[test])
    print(accuracy_score(true_labels, predictions))
    print(confusion_matrix(true_labels, predictions))
    pred_prob = svm_model.predict_proba(X_trn.iloc[test])
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label=1)
    print('AUC = ', round(metrics.auc(fpr, tpr),6))

# Convolutional neural network training
for train, test in kfold.split(X_trn, y_trn):
    model = cnn_model()
    trn_new = np.asarray(X_trn.iloc[train])
    tst_new = np.asarray(X_trn.iloc[test])   
    ## evaluate the model
    model.fit(trn_new.reshape(len(trn_new),num_features,1), np_utils.to_categorical(y_trn.iloc[train],nb_classes), epochs=num_epochs, batch_size=20, verbose=0, class_weight='auto')
    #prediction
    predictions = model.predict_classes(tst_new.reshape(len(tst_new),num_features,1))
    true_labels_cv = np.asarray(y_trn.iloc[test])
    print('CV: ', confusion_matrix(true_labels_cv, predictions))