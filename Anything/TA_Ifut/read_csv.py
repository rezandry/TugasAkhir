# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:00:30 2017

@author: rezaandriyunanto
"""

from numpy import genfromtxt

#read from dataset
X = genfromtxt('../TA_Ifut/x2.csv', delimiter=',')
Y = genfromtxt('y2.csv', delimiter=',')

#Jadi ini total kolomnya ada 32

#disini ngambil kolom ke 1-6 jadi kan indexnya 0-5
x16 = X[:,[0,1,2,3,4,5]]

#disini ngambil kolom ke 7-8 jadi kan indexnya 6-7
x78 = X[:,[6,7]]