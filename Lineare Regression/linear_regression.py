#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:04:30 2018

@author: moritz
"""

import pandas as pd

"""Lebensbedingungen-Daten lesen"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' \
                 'housing/housing.data',
                 header=None,
                 sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(df.head())
