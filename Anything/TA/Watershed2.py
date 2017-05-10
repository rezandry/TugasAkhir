# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:23:01 2017

@author: rezaandriyunanto
"""

import numpy as np
import pylab
import mahotas as mh

dna = mh.imread('sampledataset.jpg')
pylab.imshow(dna)
pylab.gray()
pylab.show()

#print dna.shape
#print dna.dtype
#print dna.max()
#print dna.min()

dnaf = mh.gaussian_filter(dna, 16)
rmax = mh.regmax(dnaf)
pylab.imshow(mh.overlay(dna, rmax))