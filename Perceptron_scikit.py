#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 19:30:51 2017

@author: moritz
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import plot_decision_regions as pdr
import matplotlib.pyplot as plt

"""import iris dataset and splitting data"""
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0) #30% Testdaten, 70% Trainingsdaten

"""standardize data"""
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""train perceptron
-------------------
    trough implemented "one vs rest" method we 
    can do multiple class classification (more than two classes)"""
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

"""predict test data"""
y_pred = ppn.predict(X_test_std)
print('Fehlklassifizierte Exemplare: %d' \
      % (y_test != y_pred).sum())

"""evaluation via percentage of correct classified objects"""
print('Korrektklassifizierungsrate: %.2f' % accuracy_score(y_test, y_pred))

"""Plotten der Decision Regions"""
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
pdr.plot_decision_regions(X=X_combined_std,
                          y=y_combined,
                          classifier=ppn,
                          test_idx=range(105,150))
plt.xlabel('Länge des Blütenblattes [standardisiert]')
plt.ylabel('Breite desBlütenblattes [standardisiert]')
plt.legend (loc='upper left')
plt.show()


        
        