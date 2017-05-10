# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:19:12 2017

@author: rezaandriyunanto
"""

#ini adalah svm visualize

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix

# We'll define a function to draw a nice plot of an SVM
def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

from sklearn.svm import SVC

#X = np.random.randn(20,2)
X = genfromtxt('x2.csv', delimiter=',')
#y = np.repeat([1,-1], 10)
y = genfromtxt('y2.csv', delimiter=',')
#X[y == -1] = X[y == -1] +1

plt.scatter(X[:,0], X[:,1], s=70, c=y)
plt.xlabel('X1')
plt.ylabel('X2')

svc = SVC(C=1, kernel='linear')
svc.fit(X, y)

#plot_svc(svc, X2, y2)

h=0.02 
pad=0.25

x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
# Support vectors indicated in plot by vertical lines
sv = svc.support_vectors_
plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths='1')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
print('Number of support vectors: ', svc.support_.size)