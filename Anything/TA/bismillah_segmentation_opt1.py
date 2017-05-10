# -*- coding: utf-8 -*-
"""
Created on Fri May 05 13:56:04 2017

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
h,s,v = cv2.split(hsv)

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

hsv2bgr = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(hsv2bgr,cv2.COLOR_BGR2GRAY)


# noise removal
# kernel = np.ones((3,3),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
dilasi = cv2.dilate(res2,kernel,iterations=10)
opening = cv2.morphologyEx(dilasi,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(dilasi,cv2.MORPH_CLOSE,kernel, iterations = 2)

opening = np.uint8(opening)
ret, markers = cv2.connectedComponents(opening)


#cv2.namedWindow("kmeans", cv2.WINDOW_NORMAL)
#cv2.namedWindow("opening", cv2.WINDOW_NORMAL)
#cv2.namedWindow("closing", cv2.WINDOW_NORMAL)
#cv2.imshow("kmeans", res2)
#cv2.imshow("opening", opening)
#cv2.imshow("closing", closing)
#cv2.waitKey(0)