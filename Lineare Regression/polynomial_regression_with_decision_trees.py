#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:14:14 2018

@author: moritz
"""

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression_gd import lin_regplot

"""Lebensbedingungen-Daten lesen"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' \
                 'housing/housing.data',
                 header=None,
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% der Bev. mit niedr. Sozialstatus [LSTAT]')
plt.ylabel('Preis in 1000$ [MEDV]')
plt.show()

"""Jetzt robustere Methode mit Random Forests"""