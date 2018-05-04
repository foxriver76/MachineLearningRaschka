#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:12:39 2018

@author: moritz
"""

import numpy as np
import theano
from linear_regression_theano import train_linreg, predict_linreg
import matplotlib.pyplot as plt

"""Datenmenge erzeugen"""
X_train = np.asarray([[0.0], [1.0],
                    [2.0], [3.0],
                    [4.0], [5.0],
                    [6.0], [7.0],
                    [8.0], [9.0]],
                    dtype=theano.config.floatX)

y_train = np.asarray([1.0, 1.3,
                      3.1, 2.0,
                      5.0, 6.3,
                      6.6, 7.4,
                      8.0, 9.0],
                    dtype=theano.config.floatX)


"""Linear Regression"""
costs, w = train_linreg(X_train, y_train, 
                        eta=0.001, epochs=10)
plt.plot(range(1, len(costs)+1), costs)
plt.tight_layout()
plt.xlabel('Epoche')
plt.ylabel('Straffunktion')
plt.show()

plt.scatter(X_train, 
            y_train,
            marker='s',
            s=50)
plt.plot(range(X_train.shape[0]),
         predict_linreg(X_train, w),
         color='gray',
         marker='o',
         markersize=4,
         linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()