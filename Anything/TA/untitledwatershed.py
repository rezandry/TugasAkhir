# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:40:18 2017

@author: rezaandriyunanto
"""

import cv2
# Importing opencv library
import numpy as np
import matplotlib
#  Importing NumPy,which is the fundamental package for scientific computing with Python
img = cv2.imread('sampledataset.png')
# Read the image from disk
cv2.imshow("Original image", img)  # Display image

im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##ini untuk merubah color space rgb ke hsv dengan matplotlib
imhsv = matplotlib.colors.rgb_to_hsv(im)

img_float = np.float32(imhsv)  # Convert image from unsigned 8 bit to 32 bit float
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
# Defining the criteria ( type, max_iter, epsilon )
# cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
# cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
# max_iter - An integer specifying maximum number of iterations.In this case it is 10
# epsilon - Required accuracy.In this case it is 1
k = 5  # Number of clusters
ret, label, centers = cv2.kmeans(img_float, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
# apply kmeans algorithm with random centers approach
center = np.uint8(centers)
# Convert the image from float to unsigned integer
res = center[label.flatten()]
# This will flatten the label
res2 = res.reshape(img.shape)
# Reshape the image
cv2.imshow("K Means", res2)  # Display image
cv2.imwrite("1.jpg", res2)  # Write image onto disk
# Finding the contors in the image using chain approximation
marker32 = np.int32(res2)
# converting the marker to float 32 bit
cv2.watershed(img,marker32)
# Apply watershed algorithm
m = cv2.convertScaleAbs(marker32)
ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Apply thresholding on the image to convert to binary image
thresh_inv = cv2.bitwise_not(thresh)
# Invert the thresh
res = cv2.bitwise_and(img, img, mask=thresh)
# Bitwise and with the image mask thresh
res3 = cv2.bitwise_and(img, img, mask=thresh_inv)
# Bitwise and the image with mask as threshold invert
res4 = cv2.addWeighted(res, 1, res3, 1, 0)
# Take the weighted average
# Draw the contours on the image with green color and pixel width is 1
cv2.imshow("Watershed", res4)  # Display the image
cv2.imwrite("3.jpg", res4)  # Write the image
cv2.waitKey()  # Wait for key stroke