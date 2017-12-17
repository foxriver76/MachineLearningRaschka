#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:57:59 2017

@author: moritz
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-' \
                 'databases/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1)
    
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])

"""Kreuzvalidierung mit Scikit-Learn
setzt man n_jobs auf z.B. 2 wird die Berechnung auf 2 CPUs parallel ausgeführt
(falls verfügbar). Mit n_jobs=-1 auf allen verfügbaren"""

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('KV-Korrektklassifizierungsraten-Score: \
    %s' % scores)
print('KV-Korrektklassifizierungsrate: %.3f +/- %.3f' % (
        np.mean(scores), np.std(scores))) \