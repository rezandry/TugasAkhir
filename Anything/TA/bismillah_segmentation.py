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
K = 10
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
#end KMeans

h,s,v = cv2.split(res2)

median = cv2.medianBlur(v,25)
blur = cv2.bilateralFilter(v,9,75,75)
ret, thresh = cv2.threshold(v,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
#kernel = np.ones((3,3),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
opening = cv2.morphologyEx(median,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(median,cv2.MORPH_CLOSE,kernel, iterations = 2)

# sure background area
#sure_bg = cv2.dilate(closing,kernel,iterations=5)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(closing,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
cc = markers
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(rgb_img,markers)
rgb_img[markers == -1] = [255,0,0]

from skimage.color import label2rgb
color_labels = label2rgb(markers, rgb_img, bg_label=0, alpha=0.5)
#plt.imshow(color_labels)

kernel2 = np.ones((2,2),np.uint8)
closing = cv2.morphologyEx(color_labels,cv2.MORPH_CLOSE,kernel2, iterations = 2)

im = Image.fromarray(res2)
im.save("hasil/kmeans_acceptable_12.jpg")

#im = Image.fromarray(markers)
#im.save("resultmarkers.jpg")

#im = Image.fromarray(center)
#im.save("resultkmeans.jpg")

#with open("hasil_segmentasi.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(color_labels)