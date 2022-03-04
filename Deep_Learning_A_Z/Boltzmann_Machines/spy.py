# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:15:19 2019

@author: guptad
"""

'

import pandas as pd
import numpy as np
import tensorflow as tf

movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding = 'latin-1')

#prepare training set and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t', header=None)

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t', header=None)

combined = pd.concat([training_set,test_set], axis=0)
nb_users = len(combined[0].unique()) #943
nb_movies = len(combined[1].unique()) #1682

x = training_set.copy()

def convert(data):
  new_data=[]
  for id_user in range(1, nb_users+1):
    user_data = x[x[0]==id_user]
    id_movies = user_data[1]
    id_ratings = user_data[2]
    ratings = np.zeros(nb_movies)
    ratings[id_movies-1]=id_ratings
    new_data.append(list(ratings))
  return new_data

ts = convert(x)