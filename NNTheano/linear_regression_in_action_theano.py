#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:12:39 2018

@author: moritz
"""

import numpy as np
import theano
from theano import tensor as T

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