# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:15:15 2017

@author: rezaandriyunanto
"""

import numpy as np
import cv2
import matplotlib

#---------------
preim = cv2.imread("sampledataset.jpg")
preim = cv2.cvtColor(preim, cv2.COLOR_BGR2RGB)
preim = matplotlib.colors.rgb_to_hsv(preim)

Z = np.float32(preim)
#---------------

#X = np.random.randint(25,50,(25,2))
#Y = np.random.randint(60,85,(25,2))
#Z = np.vstack((X,Y))
# convert to np.float32
#Z = np.float32(Z)
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,8,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((preim.shape))

