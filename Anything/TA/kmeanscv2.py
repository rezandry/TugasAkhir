# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 12:59:30 2017

@author: rezaandriyunanto
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('datasetpaper.jpg')
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])


hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

#insert KMeans here
Z = hsv.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

plt.imshow(res2)