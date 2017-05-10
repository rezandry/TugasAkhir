# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:13:32 2017

@author: rezaandriyunanto
"""

import cv2
import numpy as np

img = cv2.imread('sampledataset.png')
img_float = np.float32(img)  # Convert image from unsigned 8 bit to 32 bit float
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1)
# Defining the criteria ( type, max_iter, epsilon )
# cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
# cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
# max_iter - An integer specifying maximum number of iterations.In this case it is 10
# epsilon - Required accuracy.In this case it is 1
k = 5  # Number of clusters
ret, label, centers = cv2.kmeans(img_float, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
# apply kmeans algorithm with random centers approach
center = np.uint8(centers)
# Convert the image from float to unsigned integer
res = center[label.flatten()]
# This will flatten the label
res2 = res.reshape(img.shape)