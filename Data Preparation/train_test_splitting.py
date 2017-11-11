#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:02:35 2017

@author: moritz
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alkohol', 
                   'Apfelsäure', 'Asche', 
                   'Aschaelkalität', 'Magnesium', 
                   'Phenole insgesamt', 'Flavanoide',
                   'Nicht flavanoide Phenole',
                   'Tannin',
                   'Farbintensität', 'Farbe',
                   'OD280/OD315 des verdünnten Weins',
                   'Prolin']
#print('Class labels', np.unique(df_wine['Class label']))
#print(df_wine.head())
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)                    
