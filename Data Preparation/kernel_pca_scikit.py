#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:58:16 2017

@author: moritz
"""

from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=100, random_state=123)
scikit_pca = KernelPCA(n_components=2,
                       kernel='rbf', gamma=15)
X_skernpca = scikit_pca.fit_transform(X)

"""Plotten der Daten"""
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


