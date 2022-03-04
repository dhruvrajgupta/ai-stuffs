#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 02:37:16 2019

@author: dhruv
"""

# Part 1 - Data Preprocessing

# Importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



# Part 2
# Importing libraries
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#
## Initializing the ANN
#classifier = Sequential()
#
## Adding the input layer and the first hidden layer with dropout
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
#                     activation = 'relu'))
#classifier.add(Dropout(p = 0.1))
#
## Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
#                     activation = 'relu'))
#
##Adding the output layer
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
#                     activation = 'sigmoid'))
#
## Compiling the ANN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
#                   metrics = ['accuracy'])
#
## Fitting the ANN to training set
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#
## Predict the test set result
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
#
#
## Homework
#new_pred = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
#new_pred = sc.transform(new_pred)
#new_pred = classifier.predict(new_pred)
#
#
## making the confusion matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred)
# 
#df = pd.DataFrame(X)




# Part 4 - Evaluating, improving and tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                     activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                     activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10,
                             epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train,
                             cv = 10, n_jobs =-1)
mean = accuracies.mean()
variance = accuracies.std()

print(accuracies)
print(mean)


## Tuning the ANN
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#def build_classifier(optimizer):
#    classifier = Sequential()
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
#                     activation = 'relu'))
#    classifier.add(Dropout(p = 0.1))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
#                     activation = 'relu'))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
#                     activation = 'sigmoid'))
#    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', 
#                   metrics = ['accuracy'])
#    return classifier
#
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size' : [25,32],
#              'epochs' : [100,500],
#              'optimizer' : ['adam', 'rmsprop']}
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10)
#grid_search = grid_search.fit(X_train, y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_







