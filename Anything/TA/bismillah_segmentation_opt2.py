# -*- coding: utf-8 -*-
"""
Created on Mon May 08 12:36:03 2017

@author: rezaandriyunanto
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 12:59:30 2017

@author: rezaandriyunanto
"""
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('datasetpaper.jpg')
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

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

h,s,v = cv2.split(res2)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(v,cv2.MORPH_OPEN,kernel, iterations = 8)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.001*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

from skimage.color import label2rgb
color_labels = label2rgb(markers, rgb_img, bg_label=0, alpha=0.5)
