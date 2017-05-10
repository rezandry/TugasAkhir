import csv
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#read
df = pd.read_csv('result.csv', names=['1','2','3','4','5','6','7','8','9'])

#normalize
absval=abs(df)
#print absval

#temp = np.array(absval)
#temp = temp.astype(np.float)

df = df.drop('9', 1)
b, a = signal.butter(5, 10/100, 'low')
slp = signal.filtfilt(b, a, absval, padlen=0)





#visualize
plt.style.use("ggplot")
plt.plot(slp)
plt.show()

