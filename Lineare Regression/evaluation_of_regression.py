#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:31:21 2018

@author: moritz
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

"""Lebensbedingungen-Daten lesen"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' \
                 'housing/housing.data',
                 header=None,
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df.iloc[:, :-1].values
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

"""Residualdiagramm (Wert der Zielvariable wird von vorhergesagtem Wert subtrahiert)"""
plt.scatter(y_train_pred, y_train_pred-y_train, c='blue',
            marker='o', label='Trainingsdaten')
plt.scatter(y_test_pred, y_test_pred-y_test, c='lightgreen',
            marker='s', label='Testdaten')
plt.xlabel('Vorhergesagte Werte')
plt.ylabel('Residuen')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

print('Info: bei einem perfekt vorhergesagtem Modell, wären sämtliche Werte '
      'gleich 0 -> keine Abweichung zwischen pred und tatsächlichem Wert.')

"""MSE (Mean Squared Error - mittlere quadratische Abweichung) berechnen"""
