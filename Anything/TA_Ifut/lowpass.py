# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:50:19 2017

@author: rezaandriyunanto
"""

import scipy.signal.filtfilt
import scipy.signal.lfilter
import scipy
from scipy import signal
import pandas as pd

df = pd.read_csv('result1.csv', names=['1','2','3','4','5','6','7','8','9'])

#normalize
absval=abs(df)
df = df.drop('9', 1)

b, a = signal.butter(5, 0.5, 'low')
output_signal = signal.filtfilt(b, a, df)