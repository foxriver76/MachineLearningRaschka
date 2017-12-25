#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:35:14 2017

@author: moritz
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-' \
                 'databases/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1)
    
"""SVC Pipeline"""
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 
               10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
                {'clf__C': param_range,
                 'clf__gamma': param_range,
                 'clf__kernel': ['rbf']}]

"""verschachtelte Kreuzvalidierung"""
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2,
                  n_jobs=-1)

scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('KV-Korrektklassifizierungsrate: %.3f +/- %.3f' % (
        np.mean(scores), np.std(scores)))
 
"""Jetzt Vergleich mit DecisionTree Kreuzvalidierung"""
gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        param_grid=[
                {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                scoring='accuracy',
                cv=5)
scores = cross_val_score(gs,
                         X_train,
                         y_train,
                         scoring='accuracy',
                         cv=2)
print('KV-Korrektklassifizierungsrate: %.3f +/- %.3f' % (
        np.mean(scores), np.std(scores)))
