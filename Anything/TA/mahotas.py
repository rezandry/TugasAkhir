import numpy as np
import cv2
from matplotlib import pyplot as plt
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

#here i need hsv to RGB
RGB = cv2.cvtColor(res2,cv2.COLOR_HSV2RGB)
#then i need RGB to gray


gray = cv2.cvtColor(RGB,cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(RGB,markers)
RGB[markers == -1] = [255,0,0]

cv2.imshow('watershed',RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()