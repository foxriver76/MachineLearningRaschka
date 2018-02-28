#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:23:46 2018

@author: moritz
"""
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
import pandas as pd



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

"""Regulasierung gegen Overfitting"""
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)

lasso.fit(X_train, y_train)

"""für weiteres Buch Seite 299f."""
