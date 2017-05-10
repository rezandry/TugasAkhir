# -*- coding: utf-8 -*-
"""
Created on Tue May 09 19:24:51 2017

@author: rezaandriyunanto
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 08 15:46:28 2017

@author: rezaandriyunanto
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 12:59:30 2017

@author: rezaandriyunanto
"""
import csv
import numpy as np
from numpy import array
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import scipy
from skimage.color import label2rgb
from skimage.measure import label

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
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
#end KMeans

h,s,v = cv2.split(res2)

median = cv2.medianBlur(v,25)

z = []
dims = median.shape
for x in range(0,dims[0]):
    for y in range (0,dims[1]):
        z.append(median[x][y])


(n, bins, patches) = plt.hist(z, bins=20)

histcv = cv2.calcHist([median],[0],None,[256],[0,256])


plt.hist(median.ravel(),256,[0,256])
plt.title('Histogram of Hue')
plt.xlabel('variable X (bin size = 8)')
plt.ylabel('count')

plt.show()