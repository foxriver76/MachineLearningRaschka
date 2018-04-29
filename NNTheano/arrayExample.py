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