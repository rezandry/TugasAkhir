# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:20:15 2017

@author: rezaandriyunanto
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img2 = cv2.imread('datasetpaper.jpg')
img = cv2.imread('hasil/kmeans_acceptable_10.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img2,markers)
img2[markers == -1] = [255,0,0]

im = Image.fromarray(img2)
im.save("hasil/watershed_try.jpg")