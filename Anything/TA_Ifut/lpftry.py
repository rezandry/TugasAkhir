# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:47:03 2017

@author: rezaandriyunanto
"""

import csv
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import obspy
from scipy.signal import hilbert, chirp
#read
df = pd.read_csv('result2.csv', names=['1','2','3','4','5','6','7','8','9'])

#normalize
df[df<0]=abs(df)
df[df<20]=0
fs = 20
df = df.drop('9', 1)
df = df.drop('8', 1)
df = df.drop('7', 1)
df = df.drop('6', 1)
df = df.drop('5', 1)
df = df.drop('4', 1)
df = df.drop('3', 1)
df = df.drop('1', 1)

b, a = signal.butter(150, 0.5, 'lowpass')
slp = signal.filtfilt(b, a, df, padlen=0)

analytic_signal = hilbert(slp)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs


#plt.style.use("ggplot")
#plt.plot(slp)
#plt.show()

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(slp, label='signal')
ax0.plot(amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(analytic_signal)
ax1.set_xlabel("time in seconds")
plt.plot(df)
plt.show()