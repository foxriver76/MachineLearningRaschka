#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 10:31:47 2018

@author: moritz
"""

import theano
from theano import tensor as T
import numpy as np

"""2x3 Matrix erzeugen und Spaltensumme berechnen
Grundsätzlich drei Schritte Init, kompilieren und ausführen"""

# Initialisieren
# Im 64-Bit Modus muss man fmatrix statt dmatrix verwenden
x = T.matrix(name='x')
x_sum = T.sum(x, axis=0)

# Kompilieren
calc_sum = theano.function(inputs=[x], outputs=x_sum)

# Ausführen (Python-Liste)
ary = [[1, 2, 3], [1, 2, 3]]
print('Spaltensumme:', calc_sum(ary))

# Ausführen (Numpy-Array)
ary = np.array([[1, 2, 3], [1, 2, 3]],
               dtype=theano.config.floatX)
print('Spaltensumme:', calc_sum(ary))

"""Speicherverwaltung Beispiel"""

# Initialisieren
X = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]],
                             dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# Kompilieren
net_input = theano.function(inputs=[x],
                            updates=update,
                            outputs=z)

# Ausführen
data = np.array([[1, 2, 3]],
                dtype=theano.config.floatX)

for i in range(5):
    print('z%d:' % i, net_input(data))
    
"""Durch givens-Parameter Anzahl der Übertragungen zwischen CPU und 
GPU verringern"""

# Initialisieren
data = np.array([[1, 2, 3]],
                dtype=theano.config.floatX)
x = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]],
                             dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# Kompilieren
net_input = theano.function(inputs=[],
                            updates=update,
                            givens={x: data}, # givens ist ein Dictionary
                            outputs=z)

# Ausführen
for i in range(5):
    print('z:', net_input())
    
