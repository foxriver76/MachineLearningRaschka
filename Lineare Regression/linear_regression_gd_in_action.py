#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:08:55 2018

@author: moritz
"""

from linear_regression_gd import LinearRegressionGD, lin_regplot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

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

"""Standardisieren für bessere Konvergenz"""
sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

"""Konvergenz plotten -> Epochen vs Summe quad. Abweichungen"""
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('Summe quadrierter Abweichungen')
plt.xlabel('Epochen')
plt.show()

"""Plotten des Streudiagramms zur Beurteilung"""
lin_regplot(X_std, y_std, lr)
plt.xlabel('Durchschnittliche Anzahl der Zimmer [RM] (standardisiert)')
plt.ylabel('Preis in 1000$ [MEDV] (standardisiert)')
plt.show()

"""Entstandardisieren von Daten"""
num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print('Preis in 1000$s: %.3f' % sc_y.inverse_transform(price_std))

"""Gewichtungen von y-Achsenabschnitten müssen nicht angepasst
werden solange Variablen standardisiert sind, siehe:"""
print('Steigung: %.3f' % lr.w_[1])
print('Achsenabschnitt %.3f' % lr.w_[0])