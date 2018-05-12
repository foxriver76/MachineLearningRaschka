#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:14:43 2018

@author: moritz
"""
from load_mnist import load_mnist
import theano
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np

X_train, y_train = load_mnist('Data', kind='train')
print('Training rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('Data', kind='t10k')
print('Test rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

"""Transform MNIST-data to 32bit"""
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

"""Klassenzahlen in One-Hot-Format bringen"""
print('Die ersten 3:', y_train[:3])

y_train_ohe = np_utils.to_categorical(y_train)
print('\nDie ersten 3 (One-Hot):\n', y_train_ohe[:3])

"""Implementing Neural Net with Keras"""
np.random.seed(1)

"""Model für Feedforward initialisieren"""
model = Sequential()
"""Input Layer"""
model.add(Dense(input_dim=X_train.shape[1],
                output_dim=50,
                init='uniform',
                activation='tanh'))

"""Hidden Layer"""
model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'))
"""Output Layer"""
model.add(Dense(input_dim=50,
                output_dim=y_train_ohe.shape[1],
                init='uniform',
                activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)