#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:06:55 2018

@author: moritz
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


"""Random Daten erzeugen"""
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

"""Distanzmatrix erzeugen"""
row_dist = pd.DataFrame(squareform(
        pdist(df, metric='euclidean')),
        columns=labels, index=labels)
print(row_dist)

"""Complete Linkage Agglomerative Clustering"""