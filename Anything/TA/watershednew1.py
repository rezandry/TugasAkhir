# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:07:56 2017

@author: rezaandriyunanto
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('datasetpaper.jpg')
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
dilation = cv2.dilate(thresh,kernel,iterations = 1)

plt.subplot(131),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(thresh, 'gray')
plt.title("Otus's binary threshold"), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(dilation, 'gray')
plt.title("Closing Morph"), plt.xticks([]), plt.yticks([])
plt.show()    