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

def predict(self, X):
    """Klassenbezeichnung für X vorhersagen.

    Parameter
    ---------
    X : {Array-artige, dünnbesetzte Matrix},
        Shape = [n_samples, n_features]
        Matrix der Trainingsobjekte
    
    Rückgabewert
    ------------
    maj_vote : Array-artig, shape = [n_samples]
        Vorhergesagte Klassenbezeichnung
    """
    
    if self.vote == 'probability':
        maj_vote = np.argmax(self.predict_proba(X),
                             axis=1)
    else: # vote is 'classlabel'
            
        # Resultate der clf.predict-Aufrufe ermitteln
        predictions = np.asarray([clf.predict(X)
            for clf in 
            self.classifiers_]).T
        maj_vote = np.apply_along_axis(
                lambda x:
                    np.argmax(np.bincount(x,
                                          weights=self.weights)),
                                          axis=1,
                                          arr=predictions)
    maj_vote = self.labelenc_.inverse_transform(maj_vote)
    return maj_vote
        
def predict_proba(self, X):
    """Klassenwahrscheinlichkeiten für X vorhersagen
    
    Parameter
    ---------
    X: {Array-artige, dünnbesetzte Matrix},
        Shape = [n_samples, n_Features]
        Trainingsvektoren; n_samples ist die Anzahl der
        Objekte und n_features die Anzahl der Merkmale
        
    Rückgabewert
    ------------
    avg_proba: Array-artig
        shape = [n_samples, n_classes]
        Gewichtete durchschnittliche Wahrscheinlichkeit
        für jede Klasse pro Objekt
    """
    
    probas = np.asarray([clf.predict_proba(X)
        for clf in self.classifiers_])
    avg_proba = np.average(probas, 
                           axis=0, weights=self.weights)
    return avg_proba

def get_params(self, deep=True):
    """Klassifizierer-Parameternamen für Rastersuche"""                          
    
    if not deep:
        return super(MajorityVoteClassifier, 
                     self).get_params(deep=False)
    else:
        out = self.named_classifiers.copy()
        for name, step in\
                six.iteritems(self.named_classifiers):
            for key, value in six.iteritems(
                    step.get_params(deep=True)):
                out['%s__%s' % (name, key)] = value
        return out