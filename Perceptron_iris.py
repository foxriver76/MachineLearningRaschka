#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:24:11 2017

@author: moritz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Perceptron as percept
from matplotlib.colors import ListedColormap

"""Reading the data"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

"""splitting df in label vector and the data"""
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

"""scatter plotting the data"""
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1],
            color='blue', marker='x', label='versicolor')
plt.xlabel('Länge des Kelchblattes [cm]')
plt.ylabel('Länge des Blütenblattes [cm]')
plt.legend(loc='upper left')
plt.show


"""new Perceptron"""
ppn = percept.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
         marker='o')
plt.xlabel('Epochen/Durchläufe')
plt.ylabel('Anzahl der Updates')
plt.show


