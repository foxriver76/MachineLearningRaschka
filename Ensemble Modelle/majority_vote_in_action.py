#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:18:27 2018

@author: moritz
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mehrheitsentscheidungsklassifizierer import MajorityVoteClassifier

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test =\
     train_test_split(X, y,
                      test_size=0.5, 
                      random_state=1)
     
"""Logitische Regression, DecisionTree, k-nearest Neighbour trainieren 
und vergleichen dann zu Ensemble kombinieren"""