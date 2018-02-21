#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:52:02 2018

@author: moritz
"""

from sklearn.linear_model import RANSACRegressor, LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""Lebensbedingungen-Daten lesen"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' \
                 'housing/housing.data',
                 header=None,
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['RM']].values
y = df['MEDV'].values

"""RANSAC Regressor um Outlier weniger Gewicht zu geben bei einer linearen Regression"""
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         loss='absolute_loss',
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X, y)

"""Visualisierung der Inliners und Outliers"""
inliner_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inliner_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inliner_mask], y[inliner_mask],
            c='blue', marker='o', label='Inliner')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='lightgreen', marker='s', label='Outlier')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Durchschnittliche Anzahl der Zimmer [RM]')
plt.ylabel('Preis in 1000$ [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Steigung: %.3f' % ransac.estimator_.coef_[0])
print('Achsenabschnitt: %.3f' % ransac.estimator_.intercept_)