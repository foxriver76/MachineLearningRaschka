#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:06:23 2017

@author: moritz
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from plot_decision_regions import plot_decision_regions
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

"""train logistic regression classifier"""
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

"""Plotten der Decision Regions"""
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, 
                      y_combined, classifier=lr,
                      test_idx=range(105,150))
plt.xlabel('Länge des Blütenblattes [standardisiert]')
plt.ylabel('Breite des Blütenblattes [standardisiert]')
plt.legend(loc='upper left')
plt.show()
