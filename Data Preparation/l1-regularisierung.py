#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:11:15 2017

@author: moritz
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alkohol', 
                   'Apfelsäure', 'Asche', 
                   'Aschaelkalität', 'Magnesium', 
                   'Phenole insgesamt', 'Flavanoide',
                   'Nicht flavanoide Phenole',
                   'Tannin',
                   'Farbintensität', 'Farbe',
                   'OD280/OD315 des verdünnten Weins',
                   'Prolin']
#print('Class labels', np.unique(df_wine['Class label']))
#print(df_wine.head())
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  

"""Normalize Data between 0 and 1"""
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

"""Standardize Data with mean=0 and std=1"""#
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

"""Ab hier L1-Regularisierung"""
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)

print('Korrektklassifizierungsrate Training:',
      lr.score(X_train_std, y_train))

print('Korrektklassifizierungsrate Test:',
      lr.score(X_test_std, y_test))

"""da wir mehr als 2 Klassen haben wir standardmäßig OvR benutzt - erste Spalte 
von intercept_: Klasse 1 als positiv - die anderen negativ, 2. Spalte 2. Klasse
positiv rest negativ, 3. Spalte 3. Klasse positiv, rest negativ""" 
#print(lr.intercept_)

"""lr.coef_ enthält 3 Zeilen und die 13 Merkmale als Spalten - diese
werden mit den Merkmalen multipliziert um Nettoeingabe zu berechnen"""
#print(lr.coef_)
