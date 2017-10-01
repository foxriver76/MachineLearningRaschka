#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 19:15:52 2017

@author: moritz
"""
import matplotlib.pyplot as plt
import AdalineGD as ada
import numpy as np
import pandas as pd

"""Reading the data"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

"""splitting df in label vector and the data"""
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

"""Plotting error per iterations"""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4)) # does two subplots in one row with given size
ada1 = ada.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
  np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochen')
ax[0].set_ylabel('log(Summe quadrierter Abweichungen)')
ax[0].set_title('Adaline - Lernrate 0.01')
ada2 = ada.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada1.cost_) + 1),
  np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochen')
ax[1].set_ylabel('log(Summe quadrierter Abweichungen)')
ax[1].set_title('Adaline - Lernrate 0.0001')
plt.show()