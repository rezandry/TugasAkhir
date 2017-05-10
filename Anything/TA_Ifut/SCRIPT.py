# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:09:11 2017

@author: rezaandriyunanto
"""
import csv
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

#read
df = pd.read_csv('result.csv', names=['1','2','3','4','5','6','7','8','9'])

#normalize
df[df<0]=abs(df)
df[df<20]=0

#visualize
plt.style.use("ggplot")

#
fig, axes = plt.subplots(nrows=3, ncols=3)
#df.plot(ax=axes[0,0],y='1',label='channel1')
#df.plot(ax=axes[0,1],y='2',label='channel2')

#show visualize
#df.plot(y='1')
#flag = list()
#dimension of array
dims = df.shape

#find node have value or not
#for x in range(0,dims[0]):
#    y=1        
#    while y<9 :
#        if df[str(y)][x]>0:
#            flag.append(50)
#            break
#        else: 
#            flag.append(0)
#            y += 1




#find the MAV
resultrms = list()
resultmav = list()
resultssi = list()
resultvar = list()
se = list()
y=1
n=0
start = 0
inistart = 0
end = 0
mav = 0
rms = 0
ssi = 0
var = 0

while y<9 :
    for x in range(0,dims[0]):
        if df[str(y)][x] > 0 and df[str(y)][x+1] !=0:        
            n += 1
        elif df[str(y)][x] > 0 and df[str(y)][x+1] == 0:
            n += 1            
            if(n == 1) :
                df[str(y)][x] = 0
            n = 0
    y += 1

y=1
n=0

while y<9 :
    for x in range(0,dims[0]):
        if df[str(y)][x] > 0 and df[str(y)][x+1] !=0:        
            mav = mav + df[str(y)][x]
            rms = rms + df[str(y)][x]**2
            ssi = ssi + df[str(y)][x]**2
            var = var + df[str(y)][x]**2
            n += 1
            if start == 0 :
                inistart = x
                start += 1
        elif df[str(y)][x] > 0 and df[str(y)][x+1] == 0:
            n += 1            
            if(n == 1) :
                df[str(y)][x] = 0
            mav = mav + df[str(y)][x]
            rms = rms + df[str(y)][x]**2
            ssi = ssi + df[str(y)][x]**2
            var = var + df[str(y)][x]**2            
            
            hasilmav = mav/n
            hasilrms = math.sqrt(rms/n)
            hasilssi = ssi
            hasilvar = var/(n-1)
            resultmav.append(y)
            resultmav.append(hasilmav)
            resultrms.append(y)
            resultrms.append(hasilrms)
            resultssi.append(y)
            resultssi.append(hasilssi)
            resultvar.append(y)
            resultvar.append(hasilvar)
            
                        
            se.append(x)
            print (y)
            print (inistart)            
            print (x)
            print ("--------------")
            n = 0
            mav = 0
            rms = 0
            ssi = 0
            var = 0
            start = 0
    y += 1
            
            
 
#convert to file
csvfile = 'listiden.csv'
res = df
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in res:
        writer.writerow([val])