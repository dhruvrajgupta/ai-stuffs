# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:05:54 2019

@author: guptad
"""

#Data preprocessing
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

dataset_train = np.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


#RNN
