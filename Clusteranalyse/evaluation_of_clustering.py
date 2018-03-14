#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:05:36 2018

@author: moritz
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


"""Create Sample Data"""
X, y = make_blobs(n_samples=150,
                  centers=3,
                  n_features=2,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

"""k_means auf die Daten anwenden"""
km = KMeans(n_clusters=3, 
             init='random',
             n_init=10,
             max_iter=300,
             tol=1e-04,
             random_state=0)

y_km = km.fit_predict(X)


"""Summe quadrierter Abweichungen zur Evaluation"""
print('Verzerrung: % .2f' % km.inertia_)

"""Ellenbogenkriterium --> Plotten der Verzerrung für unterschiedliche 
k-Werte"""
distortions = []
for i in range(1, 11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Summe quadrierter Abweichungen - Verzerrung')
plt.show()