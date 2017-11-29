#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:08:41 2017

@author: moritz
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)    
X_test_std = sc.transform(X_test)

"""Mittelwertvektoren berechnen"""
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(
            X_train_std[y_train==label], axis=0))
    print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    
"""Einzelne Streumatrizen berechnen"""
d = 13 # Anzahl der Merkmale
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
print('Streumatrix innerhalb der Klasse: %sx%s'
      % (S_W.shape[0], S_W.shape[1]))

"""Verteilung der Klassen/Labels ausgeben"""
print('Verteilung der Klassenbezeichnungen: %s'
      % np.bincount(y_train)[1:])

"""Gesamte Streumatrix berechnen"""
d = 13 # Anzahl der Merkmale
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('Skalierte Streumatrix innerhalb der Klasse: %sx%s'
      % (S_W.shape[0], S_W.shape[1])) #skalierte Streumatrix = Kovarianzmatrix

"""Streumatrix für Streuung zwischen Klassen berechnen"""
mean_overall = np.mean(X_train_std, axis=0)
d = 13 # Anzahl Merkmale
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot(
            (mean_vec - mean_overall).T)
print('Streumatrix für die Streuung zwischen Klassen: %sx%s'
      % (S_B.shape[0], S_B.shape[1]))
    
"""Eigenwertproblem der Matrix lösen"""
eigen_vals, eigen_vecs = \
    np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs,
                     key=lambda k:k[0], reverse=True)
print('Eigenwerte in absteigender Reihenfolge:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])