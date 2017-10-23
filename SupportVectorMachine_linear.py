#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:10:13 2017

@author: moritz
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

"""Plotten der Decision Regions"""
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                          y=y_combined,
                          classifier=svm,
                          test_idx=range(105,150))
plt.xlabel('Länge des Blütenblattes [standardisiert]')
plt.ylabel('Breite des Blütenblattes [standardisiert]')
plt.legend(loc='upper left')
plt.show()