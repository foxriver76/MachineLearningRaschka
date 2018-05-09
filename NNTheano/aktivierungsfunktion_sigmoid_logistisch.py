#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:09:34 2018

@author: moritz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

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

"""Wahrscheinlichkeit der Klassenzugehörigkeit mit softmax Funktion abschätzen"""
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_activation(X, w):
    z = net_input(X, w)
    return softmax(z)

y_probas = softmax(Z)
print('Probabilities:\n', y_probas)

print('Sum of probabilities', y_probas.sum())

y_class = np.argmax(Z, axis=0)
print('Predicted class', y_class)

"""Tangens hyperbolicus Aktivierungsfunktion [-1, 1] statt [0, 1] wie Logistik"""
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

# alternatives:
# from scipy.special import expit
# log_act = expit(z)
# tanh_act = np.tanh(z)

plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle='--')
plt.axhline(0.5, color='black', linestyle='--')
plt.axhline(0, color='black', linestyle='--')
plt.axhline(-1, color='black', linestyle='--')

plt.plot(z, tanh_act,
         linewidth=2,
         color='black',
         label='tanh')
plt.plot(z, log_act,
         linewidth=2,
         color='lightgreen',
         label='logistic')

plt.legend(loc='lower right')
# plt.tight_layout()
# plt.savefig('./figures/activation.png', dpi=300)
plt.show()

"""tanh und logistische Funktion sind schon vor implementiert"""
tanh_act = np.tanh(z)
log_act = expit(z)