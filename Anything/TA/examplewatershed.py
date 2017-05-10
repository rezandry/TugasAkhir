# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 01:39:09 2017

@author: rezaandriyunanto
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('datasetpaper.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

unknown = opening

# Marker labelling
ret, markers = cv2.connectedComponents(opening)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.imshow(img)