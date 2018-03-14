#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:14:55 2018

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

"""Plotten der Daten"""
plt.scatter(X[:, 0], X[:, 1],
            c='black',
            marker='o',
            s=50)
plt.grid()
plt.show()


"""k_means auf die Daten anwenden"""
km = KMeans(n_clusters=3, 
             init='random',
             n_init=10,
             max_iter=300,
             tol=1e-04,
             random_state=0)

y_km = km.fit_predict(X)

"""Plotten der Ergebnisse"""
plt.scatter(X[y_km==0, 0], X[y_km==0, 1],
            c='b',
            marker='s',
            label='Cluster 1',
            s=50)
plt.scatter(X[y_km==1, 0], X[y_km==1, 1],
            c='r',
            marker='o',
            label='Cluster 2',
            s=50)
plt.scatter(X[y_km==2, 0], X[y_km==2, 1],
            c='lightgreen',
            marker='v',
            label='cluster3',
            s=50)
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='orange',
            label='Zentroiden')
plt.legend()
plt.grid()
plt.show()


"""Summe quadrierter Abweichungen zur Evaluation"""
print('Verzerrung: % .2f' % km.inertia_)
