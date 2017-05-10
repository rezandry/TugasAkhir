import cv2
import matplotlib
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
##aktifkan ini apabila ingin menampilkan gambar hasil convert ke hsv
#import matplotlib.pyplot as plt

##library yang ini masih belum kepakai
#import matplotlib.image as mpimg
#import numpy as np

##ini untuk membaca gambar dengan opencv 
#default imread dari opencv adalah dalam color space BGR 
preim = cv2.imread("sampledataset.jpg")
#convert BGR colorspace ke RGB colorspace
im = cv2.cvtColor(preim, cv2.COLOR_BGR2RGB)

##ini untuk merubah color space rgb ke hsv dengan matplotlib
imhsv = matplotlib.colors.rgb_to_hsv(im)

##ini untuk melihat hasil convert ke hsv
#imgplot2 = plt.imshow(imhsv)
dims = imhsv.shape

##Fitur 1 : Average hue for the whole image
##inisialisasi untuk sum_hue untuk menyimpan sum of hue
sum_hue = 0
##ini masuk ke fitur pertama, yakni average hue for whole image
for x in range(0,dims[0]):
    for y in range (0,dims[1]):
        sum_hue = sum_hue + imhsv[x][y][0]
##hue saturation saved in imhsv[x][y][0]
f1 = sum_hue/(dims[0]*dims[1])

##Fitur 2 : Average saturation for the whole image
##inisialisasi untuk sum_saturation untuk menyimpan sum of saturation
sum_sat = 0
##ini masuk ke fitur pertama, yakni average saturation for whole image
for x in range(0,dims[0]):
    for y in range (0,dims[1]):
        sum_sat = sum_sat + imhsv[x][y][1]
##hue saturation saved in imhsv[x][y][0]
f2 = sum_sat/(dims[0]*dims[1])

##here want to make histogram from hue
z = list()

for x in range(0,dims[0]):
    for y in range (0,dims[1]):
        z.append(imhsv[x][y][0])


(n, bins, patches) = plt.hist(z, bins=20)
plt.title('Histogram of Hue')
plt.xlabel('variable X (bin size = 20)')
plt.ylabel('count')

plt.show()

##Feature 3 : Number of Quantized Hues
#Equation number of hue that more than c * q
#Variable c = constanta 0 < c <=1
c = 0.1
#Variable q = largest value of histogram

#Find q
q = 0
for x in range(0,20):
    if q < n[x]: 
        q = n[x]

#Find number of Quantized Hues
f3 = 0
for x in range(0,20):
    if n[x] > c*q:
        f3 += 1

##Feature 4-23 :
f4 = n[0]
f5 = n[1]
f6 = n[2]
f7 = n[3]
f8 = n[4]
f9 = n[5]
f10 = n[6]
f11 = n[7]
f12 = n[8]
f13 = n[9]
f14 = n[10]
f15 = n[11]
f16 = n[12]
f17 = n[13]
f18 = n[14]
f19 = n[15]
f20 = n[16]
f21 = n[17]
f22 = n[18]
f23 = n[19]

##Lets Go to Feature 24
#lets see, first, we must know k-means



















