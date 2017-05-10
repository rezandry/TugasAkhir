# -*- coding: utf-8 -*-
"""
Created on Mon May 08 14:59:40 2017

@author: rezaandriyunanto
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2

from skimage.morphology import watershed
from skimage.feature import peak_local_max


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

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(v)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=v)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=v)
fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(v, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()