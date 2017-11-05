#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:23:57 2017

@author: moritz
"""
import AdalineSGD as ada
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_decision_regions as pdr

"""Reading the data"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)

"""splitting df in label vector and the data"""
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

"""Standardization of data to mean = 0 std = 1"""
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

"""Training Adaline with Stochastical Gradient Descent"""
adaline = ada.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
adaline.fit(X_std, y)
pdr.plot_decision_regions(X_std, y, classifier=adaline)
plt.title('Adaline - Stochastisches Gradientenabstiegsverfahren')
plt.xlabel('Länge des Kelchblattes [standardisiert]')
plt.ylabel('Länge des Blütenblattes [standardisiert]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(adaline.cost_) + 1),
         adaline.cost_, marker='o')
plt.xlabel('Epochen')
plt.ylabel('Durchschnittswert der Strafffunktion')
plt.show()

