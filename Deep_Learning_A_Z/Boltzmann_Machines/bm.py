# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:21:23 2019

@author: guptad
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


    
#impoeting the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding = 'latin-1')
