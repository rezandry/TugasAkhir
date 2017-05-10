# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:51:18 2017

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