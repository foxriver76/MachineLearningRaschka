#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 19:30:51 2017

@author: moritz
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

"""import iris dataset and splitting data"""
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0) #30% Testdaten, 70% Trainingsdaten

"""standardize data"""
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""train perceptron
-------------------
    trough implemented "one vs rest" method we 
    can do multiple class classification (more than two classes)"""
    
