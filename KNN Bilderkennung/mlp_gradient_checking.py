#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 08:49:21 2018

@author: moritz
"""

from readData import load_mnist
from neuronales_netz import MLPGradientCheck

"""Daten einlesen"""
X_train, y_train = load_mnist('Data', kind='train')

X_test, y_test = load_mnist('Data', kind='t10k')

"""Gradient checking, Minibatch=1, Regularisierung deaktivieren"""
nn_check = MLPGradientCheck(n_output=10,
                            n_features=X_train.shape[1],
                            n_hidden=10,
                            l2=0.0,
                            l1=0.0,
                            epochs=10,
                            eta=0.001,
                            alpha=0.0,
                            decrease_const=0.0,
                            minibatches=1,
                            shuffle=False,
                            random_state=1)

"""Sehr rechenaufwendig, nur für Debugging geeignet, deshalb hier nur 5 Trainingsobjekte"""
nn_check.fit(X_train[:5], y_train[:5], print_progress=False)

