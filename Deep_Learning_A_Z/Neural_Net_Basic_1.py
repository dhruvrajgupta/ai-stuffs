# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:45:32 2019

@author: guptad
"""

import numpy as np

def sigmoid(X,deriv=False):
    if(deriv==True):
        return X*(1-X)
    return 1/(1+np.exp(-X))

X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
#print("Shape of X : ",X.shape)

Y = np.array([[0,1,1,0]]).T
#print("Shape of Y : ",Y.shape)

np.random.seed(1)

syn0 = np.random.random((3,1))
#print("syn0 : \n",syn0)
#print("\n")

for iter in range(10000):
    #Forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    
    #find error
    l1_error = Y-l1
    #print("Error : \n",l1_error)
    
    l1_delta = l1_error*sigmoid(l1,True)

    #print("Delta : \n",l1_delta)
    
    syn0 += np.dot(l0.T,l1_delta)
    