#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:35:13 2018

@author: moritz
"""

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd


"""Random Daten erzeugen"""
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

"""Hierarchisches Clustering aus Scikit nutzen"""
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print('Clusterbeziehungen: %s' % labels)
