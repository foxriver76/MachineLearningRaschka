#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 08:15:13 2017

@author: moritz
"""

from kernel_pca import rbf_kernel
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.show()

"""Daten nicht linear trennbar, wir müssen Sie mit Kernel PCA trennen damit 
wir sie als Eingabe für einen linearen Klassifizierer nutzen können.
Doch zuerst betrachten wir normale PCA"""
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
          color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
          color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,
          color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,
          color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylin([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

"""Jetzt Kernel_PCA"""
