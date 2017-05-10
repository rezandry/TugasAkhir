# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:29:04 2017

@author: rezaandriyunanto
"""

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
import cv2
import matplotlib

# data generation
# data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))

preim = cv2.imread("sampledataset.jpg")
#convert BGR colorspace ke RGB colorspace
im = cv2.cvtColor(preim, cv2.COLOR_BGR2RGB) 
##ini untuk merubah color space rgb ke hsv dengan matplotlib
data = matplotlib.colors.rgb_to_hsv(im)

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,2)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()