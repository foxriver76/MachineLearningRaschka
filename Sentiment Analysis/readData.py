#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:16:21 2018

@author: moritz
"""

import pyprind 
import pandas as pd
import os
import numpy as np

"""Einzelnen Textdokumente in eine Datei mergen"""
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg': 0}
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file),
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

"""Sortierte Datensammlung durchmsichen"""
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)

"""Prüfen ob Datensätze richtig gespeichert wurden"""
df = pd.read_csv('./movie_data.csv')
print(df.head(3))