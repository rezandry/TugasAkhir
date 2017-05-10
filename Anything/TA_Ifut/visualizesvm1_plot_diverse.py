# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:29:31 2017

@author: rezaandriyunanto
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:24:29 2017

@author: rezaandriyunanto
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

# We’ll define a function to draw a nice plot of an SVM

#X = np.random.randn(200,2)
#X[:100] = X[:100] +2
#X[101:150] = X[101:150] -2
#y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])
X = genfromtxt('x2.csv', delimiter=',')
y = genfromtxt('y2.csv', delimiter=',')


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)



plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel("X1")
plt.ylabel("X2")

