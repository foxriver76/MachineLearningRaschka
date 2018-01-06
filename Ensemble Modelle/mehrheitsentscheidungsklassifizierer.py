#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 11:14:44 2018

@author: moritz
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.prepprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.pipeline import _name_estimators
import operator



print(np.argmax(np.bincount([0, 0 , 1],
          weights=[0.2, 0.2, 0.6])))

ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])

p =  np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
print(p)
print(np.argmax(p))

"""Mehrheitsklassifizierer implementieren"""

class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    """Ein Mehrheitsentscheidungsklassifizierer als Ensemble
    
    Parameter
    ---------
    classifiers : array-like, shape = [n_classifiers]
        Verschiedene Klassifizierer des Ensemble
    
    vote: str, {'classlabel', 'probability'}
        Standard: 'classlabel'
        Falss 'classlabel', beruht die Vorhersage auf dem 
        argmax-Wert der Klassenbezeichnungen.
        Falls 'probability', wird der argmax-Wert aus der Summe der
        Wahrscheinlichkeiten zur Vorhersage der Klassen-
        bezeichnung verwendet (bei kalibrierten Klassifizierern 
                               empfehlenswert).
    
    weights : array-like, shape = [n_classifiers]
        Optional, Standard: Keine
        Falls eine Liste mit 'int'- oder 'float'-Werten
        übergeben wird, werden die Klassifizierer ihrerer
        Bedeutung nach gewichtet. Falls 'weights=None',
        werden alle gleich gewichtet.
        
    """
    
    def __init__(self, classifiers,
                 vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for
                                  key, value in
                                  _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        
    def fit(self, X, y):
        """Klassifizierer anpassen
        
        Parameter
        ---------
        X : {Array-artige, dünnbesetzte Matrix},
            shape = [n_samples, n_features]
            Matrix der Trainingsobjekte
            
        y : Array-artig,  shape = [n_samples]
            Vektor der Zielklassenbezeichnungen
            
        Rückgabewert
        ------------
        self : object 
        
        """
        
        # LabelEncoder verwenden, um zu gewährleisten, dass
        # die Klassenbezeichnungen bei 0 beginnen; wichtig 
        # für den np.argmax-Aufruf in self.predict
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,
                              self.labelenc_.transform(y))
        self.classifiers_.append(fitted_clf)
        return self
    
        
   
    
                          