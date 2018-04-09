#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 08:35:56 2018

@author: moritz
"""

import matplotlib.pyplot as plt
from readData import load_mnist
import numpy as np

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

"""Unterschiedliche Arten von 7 ansehen"""
fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train==7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_yticks([])
ax[0].set_xticks([])
plt.tight_layout()
plt.show()

"""Speichern als csv"""
# np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
# np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
# np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
# np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')

"""Einlesen der csv"""
# X_train = np.genfromtxt('train_img.csv',
#                        dtype=int, delimiter=',')
# ...