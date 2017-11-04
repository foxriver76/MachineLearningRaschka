#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:53:27 2017

@author: moritz
"""

import pandas as pd
from io import StringIO
csv_data = '''A, B, C, D
    1.0, 2.0, 3.0, 4.0
    5.0, 6.0,, 8.0
    10.0, 11.0, 12.0,'''
    
df = pd.read_csv(StringIO(csv_data))
#print(df)
#print(df.isnull().sum()) #fehlende Werte pro Spalte
print(df.values) #Auf Numpy Array des DF zugreifen