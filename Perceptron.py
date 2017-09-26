#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:08:24 2017

@author: moritz
"""

import numpy as np

class Perceptron(object):
    """Perzeptron Klassifizierer
    
    Parameter
    ---------
    eta : float
        Lernrate (zwischen 0.0 und 1.0)
    n_iter : int
        Durchläufe der Trainingsdatenmenge
        
    Attribute
    ----------
    w_ : 1d-array
        Gewichtungen nach Anpassung
    errors_ : list
        Anzahl der Fehlklassifizierungen pro Epoche
        
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
        
    def fit(self, X, y):
        """Anpassen an die Trainingsdaten
        
        Parameter
        ---------
        X : {array-like}, shape = [n_samples, n_features]
        Trainingsvektoren, n_samples ist die Anzahl der Objekte und n_features die Anzahl der Merkmale
        y : array-like, shape = [n_samples]
        Zielwerte

        Rückgabewert
        -------------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - \
                                     self.predict())
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            return self
        
    def net_input(self, X):
        """Nettoeingabe berechnen"""
        return np.dot(X, self.w_[1:]) + self.w_[0] #dot gives scalar product of the two arrays
    
    def predict(self, X):
        """Klassenbezeichnung zurückgeben"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
    
            

    