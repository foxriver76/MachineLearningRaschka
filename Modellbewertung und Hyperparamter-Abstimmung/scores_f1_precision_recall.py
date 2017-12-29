#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:37:54 2017

@author: moritz
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

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

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 
               10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
                {'clf__C': param_range,
                 'clf__gamma': param_range,
                 'clf__kernel': ['rbf']}]

"""Scores"""
print('Genauigkeit: %.3f' % precision_score(
        y_true=y_test, y_pred=y_pred))

print('Trefferquote: %.3f' % recall_score(
        y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(
        y_true=y_test, y_pred=y_pred))

"""Merke: Klasse 1 ist in scikit immer die positive -> Alternativ
kann über make_scorer eine eigene Funktion definiert werden"""
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10)
