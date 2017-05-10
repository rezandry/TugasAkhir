# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:44:58 2017

@author: rezaandriyunanto
"""

from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from matplotlib.pyplot import plot
df = pd.read_csv('result.csv', names=['1','2','3','4','5','6','7','8','9'])

#normalize
df[df<0]=abs(df)')

input_signal = signaldata['input_signal'][0]

passband = [0.75*2/30, 5.0*2/30]
b, a = butter(5, passband, 'bandpass')

y = filtfilt(b, a, input_signal)
plot(y)