#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:25:16 2018

@author: moritz
"""
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

"""Polynomiale Regression - wenn Daten nicht in linearer Beziehung"""
"""Polynomialen Term 2. Grades hinzufügen"""
X = np.array([258.0, 270.0, 294.0,
              320.0, 342.0, 368.0,
              396.0, 446.0, 480.0,
              586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8,
              298.6, 314.2, 342.2,
              360.8, 368.0, 391.2,
              390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

"""Einfaches Regressionsmodell zu Vergleich anpassen"""
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

"""Multiples Regressionsmodell anpassen"""
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
# Ausgabe der Ergebnisse
plt.scatter(X, y, label='Trainingsdatenpunkte')
plt.plot(X_fit, y_lin_fit,
         label='Lineare Anpassung', linestyle='--')
plt.plot(X_fit, y_quad_fit,
         label='Quadratische Anpassung')
plt.legend(loc='upper left')
plt.show()

"""MSE feststellen"""
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)

print('MSE-Trainingsdaten Linear: %.3f, Quadratisch %.3f' 
      % (mean_squared_error(y, y_lin_pred),
         (mean_squared_error(y, y_quad_pred))))
print('R^2-Training Linear %.3f, Quadratisch %.3f'
      % (r2_score(y, y_lin_pred),
         r2_score(y, y_quad_pred)))
