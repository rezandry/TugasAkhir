# -*- coding: utf-8 -*-
"""
Created on Sun May 07 18:58:14 2017

@author: rezaandriyunanto
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
#sns.set(color_codes=True)

np.random.seed(sum(map(ord, "distributions")))

x = np.random.normal(size=100)
g = sns.distplot(x)