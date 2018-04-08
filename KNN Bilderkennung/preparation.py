#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 08:35:56 2018

@author: moritz
"""

import matplotlib.pyplot as plt
from readData import load_mnist

"""Daten einlesen"""
X_train, y_train = load_mnist('Data', kind='train')
print('Zeilen: %d, Spalten: %d'
      % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('Data', kind='t10k')
print('Zeilen: %d, Spalten: %d'
      % (X_test.shape[0], X_test.shape[1]))

"""Daten plotten - nachdem wir sie zurück in 28x28 Pixel gebracht haben"""
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
   
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()