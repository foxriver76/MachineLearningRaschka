#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:53:12 2017

@author: moritz
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)    
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenwerte \n%s' % eigen_vals)

"""Zur Dimensionsreduktion sind uns die größten Eigenwerte (größte Varianz) wichtig
Varianzaufklärung (cumsum) sagt uns anschließend welchen Beitrag ein EW an Summe aller EW's liefert"""
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='Individuelle Varianzaufklärung')
plt.step(range(1, 14), cum_var_exp, where='mid', 
         label='Kumulative Varianzaufkärung')
plt.ylabel('Anteil an der Varianzaufklärung')
plt.xlabel('Hauptkomponenten')
plt.legend(loc='best')
plt.show()

"""Eigenwerte & Vektoren nach Größe sortieren"""
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))]
eigen_pairs.sort(key = lambda k: k[0], reverse=True)

"""Größten beiden EVs ermitteln"""
w = np.hstack((eigen_pairs[0][1][: , np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

"""Projektionsmatrix nutzen um Daten abzubilden x'=x*W"""
#X_train_std[0].dot(w)
"""Jetzt für alle Daten"""
X_train_pca = X_train_std.dot(w)


"""Visualisierung der projizierten Daten"""
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1') 
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()   
