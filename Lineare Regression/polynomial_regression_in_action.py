#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:52:50 2018

@author: moritz
"""

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd

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
regr = LinearRegression()

"""Polynomiale Merkmale einrichten"""
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

"""Lineare Anpassung"""
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

"""Quadratische Anpassung"""
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

"""Kubische Anpassung"""
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

"""Ergebnisse ausgeben"""
plt.scatter(X, y,
            label='Trainingsdatenpunkte',
            color='lightgray')
plt.plot(X_fit, y_lin_fit,
         label='Linear (d=1), $R^2=%.2f$'
         % linear_r2,
         color='blue',
         lw=2,
         linestyle=':')
plt.plot(X_fit, y_quad_fit,
         label='Quadratisch (d=2), $R^2=%.2f$'
         % quadratic_r2,
         color='red',
         lw=2,
         linestyle='-')
plt.plot(X_fit, y_cubic_fit,
         label='Kubisch (d=3, $R^2=%.2f$'
         % cubic_r2,
         color='green',
         lw=2,
         linestyle='--')
plt.xlabel('% der Bev. mit niedr. Sozialstatus [LSTAT]')
plt.ylabel('Preis in 1000$ [MEDV]')
plt.legend(loc='upper right')
plt.show()

"""Anhang des Plots könnte man denken, dass zwischen log(MEDV) und Quadratwurzel
von LSTAT ein linearer Zusammenhang besteht, dass ist jetzt zu prüfen"""
# Merkmale transformieren