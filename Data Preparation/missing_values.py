#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:53:27 2017

@author: moritz
"""

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

"""Beim einlesen zwischen den Kommata keine Leerzeichen sonst werden diese in die String/Char Labels integriert"""
csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''
    
df = pd.read_csv(StringIO(csv_data))

"""Strategien zum entfernen von Missing Values Attributen"""
"""Anzahl fehlende Werte pro Spalte"""
#print(df.isnull().sum()) 

"""Auf Numpy Array des DF zugreifen"""
#print(df.values) 

"""Zeilen mit fehlenden Values droppen"""
#print(df.dropna())

"""Spalten mit fehlenden Values droppen"""
#print(df.dropna(axis=1))

"""Nur Zeilen löschen in denen alle Values NaN (Not a Number) sind"""
#print(df.dropna(how='all'))

"""Zeilen entfernen, die nicht mindestens 4 von NaN verschiedene Werte enthalten"""
#print(df.dropna(thresh=4))

"""Nur Zeilen löschen, in denen NaN in einer bestimmten Spalte (hier: 'C') vorkommt"""
#print(df.dropna(subset=['C']))

"""Fehlende Werte ergänzen"""
"""Mittelwerte der Merkmalsspalte einfügen"""
imr = Imputer(missing_values='NaN', 
              strategy='mean', axis=0)
"""andere Möglichkeiten sind 'median' oder 'most_frequent' wobei sich 'most_frequent für kategoriale Werte eignet""" 
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
#print(imputed_data)



