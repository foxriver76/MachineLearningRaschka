#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:22:33 2017

@author: moritz
"""

from numpy.random import seed
import numpy as np

class AdalineSGD(object):
    """Adaline Klassifizierer mit stochastischem Gradientenabstieg
    Parameter
    ---------
    eta : float
        Lernrate (zwischen 0.0 und 1.0)
    n_iter : int
        Durchläufe der Trainingsdatenmenge
        
    Attribute
    ---------
    w_ : 1d-array
        Gewichtung nach Anpassung
    errors_ : list
        Anzahl der Fehlklassifizierungen pro Epoche
    shuffle : bool (default: True)
        Falls True, werden die Trainingsdaten vor jeder neuen
        Epoche durchgemischt, um Wiederholungen zu verhindern
    random_state : int (default: None)
        Anfangswert für den Zufallsgenerator setzen
        (Durchmischen/Initialisierung der Gewichtungen)
        
    """
    def __init__(self, eta=0.01, n_iter=10,
                   shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        
        if random_state:
            seed(random_state)
            
    def fit(self, X, y):
        """Fit Trainingsdaten
        
        Parameter
        ---------
        X : {array-like}, shape = [n_samples, n_features]
            Trainingsvekotren, n_samples ist
            die Anzahl der Exemplare und 
            n_features ist die Anzahl der Merkmale
        y: array-like, shape = [n_samples]
            Zielwerte
            
        Rückgabewert
        ------------
        self : object
        
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """Anpassung an die Trainingsdaten ohne \
        Reinitialisierung der Gewichtungen """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """Trainingsdaten durchmischen"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Gewichtungen mit null initialisieren"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Adaline-Lernregel zur Aktualisierung \
                                der Gewichtungen"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Nettoeingabe berechnen"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Lineare Aktivierung berechnen"""
        return self.net_input(X)
    
    def predict(self, X):
        """Rückgabe der Klassenbezeichnung"""
        return np.where(self.activation(X) >= 0.0, 1, -1)