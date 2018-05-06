#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:09:34 2018

@author: moritz
"""

import numpy as np

"""Beispiel"""
X =  np.array([[1, 1.4, 1.5]])
w = np.array([0.0, 0.2, 0.4])

def net_input(X, w):
    z = X.dot(w)
    return z

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print('P(y=1|x) = %.3f'
      % logistic_activation(X, w)[0]) 
# Wahrscheinlichkeit das Objekt x zur positiven Klasse gehört

"""Klassenzugehörigkeit für jede Klasse in % berechnen"""