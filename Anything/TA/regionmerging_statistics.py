# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 07:48:14 2017

@author: rezaandriyunanto
"""

from scipy.misc import imread
from matplotlib import pyplot

from SRM import SRM

im = imread("hasil/kmeans_acceptable.jpg")

srm = SRM(im, 256)
segmented = srm.run()

pyplot.imshow(segmented/256)
pyplot.show()