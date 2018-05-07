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
# Man bekommt so jedoch keine Bruchteile von 100% 

"""Klassenzugehörigkeit für jede Klasse in % berechnen"""
# W : array, shape = [n_output_units, n_hidden_units+1]
#          Weight matrix for hidden layer -> output layer.
# note that first column (A[:][0] = 1) are the bias units
W = np.array([[1.1, 1.2, 1.3, 0.5],
              [0.1, 0.2, 0.4, 0.1],
              [0.2, 0.5, 2.1, 1.9]])

# A : array, shape = [n_hidden+1, n_samples]
#          Activation of hidden layer.
# note that first element (A[0][0] = 1) is for the bias units

A = np.array([[1.0],
              [0.1],
              [0.3],
              [0.7]])

# Z : array, shape = [n_output_units, n_samples]
#          Net input of output layer.

Z = W.dot(A)
y_probas = logistic(Z)
print('Probabilities:\n', y_probas)

y_class = np.argmax(Z, axis=0)
print('predicted class label: %d' % y_class[0])
