from numpy import array
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import csv
import seaborn as sns

import pandas as pd
data = pd.read_csv('aku-mav.csv', sep=';')

#Read file
#data = genfromtxt('aku-mav.csv')  
#dims = data.shape
with open ('aku-mav.csv') as cf:
    cr = csv.reader(cf, delimiter=',')
    data = [row[:8] for row in cr]
#data = array(data)
#dims = data.shape
#for x in range(0, dims[1]):
#    for y in range(0, dims[0]):
        
b = [[data[i][j] for i in range(len(data))] for j in range(8)]
ch1 = b[0]
ch1 = array(ch1).astype(np.float)
ch2 = b[1]
ch2 = array(ch2).astype(np.float)
ch3 = b[2]
ch3 = array(ch3).astype(np.float)
ch4 = b[3]
ch4 = array(ch4).astype(np.float)
ch5 = b[4]
ch5 = array(ch5).astype(np.float)
ch6 = b[5]
ch6 = array(ch6).astype(np.float)
ch7 = b[6]
ch7 = array(ch7).astype(np.float)
ch8 = b[7]
ch8 = array(ch8).astype(np.float)

#g = sns.distplot(ch1)

plt.hist(ch1, bins=10)
#(n, bins, patches) = plt.hist(ch2, bins=10)
#(n, bins, patches) = plt.hist(ch3, bins=10)
#(n, bins, patches) = plt.hist(ch4, bins=10)
#(n, bins, patches) = plt.hist(ch5, bins=10)
#(n, bins, patches) = plt.hist(ch6, bins=10)
#(n, bins, patches) = plt.hist(ch7, bins=10)
#(n, bins, patches) = plt.hist(ch8, bins=10)
