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
X_test_std = sc.fit(X_test)

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