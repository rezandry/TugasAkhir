# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:43:16 2017

@author: rezaandriyunanto
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('datasetpaper.jpg')
img2 = cv2.imread('hasil/kmeans_acceptable_12.jpg')
hsv2bgr = cv2.cvtColor(img2,cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(hsv2bgr,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
#kernel = np.ones((3,3),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=5)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

blank_image = np.zeros((1309,1000,3), np.uint8)
blank_image[:,:] = [255,255,255] 

from skimage.color import label2rgb
color_labels = label2rgb(markers, img, bg_label=0, alpha=0.5)
#plt.imshow(color_labels)