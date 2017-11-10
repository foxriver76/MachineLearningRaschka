#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:22:47 2017

@author: moritz
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

"""Dataframe mit nominalen (color), ordinalem (Größe) und numersichen (price) Feature"""
df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

"""Mappen der ordinalen Features in Nummerische"""
size_mapping = {
        'XL':3,
        'L':2,
        'M':1}

df['size'] = df['size'].map(size_mapping)

inv_size_mapping = {v: k for k, v in size_mapping.items()}

"""Umcodieren von Klassenlabels in Ganzzahlen"""
class_mapping =  {label:idx for idx, label in
                  enumerate(np.unique(df['classlabel']))}

df['classlabel'] = df['classlabel'].map(class_mapping)
 
#Rückgängig machen
inv_class_mapping = {v: k for k, v in class_mapping.items()}
#df['classlabel'] = df['classlabel'].map(inv_class_mapping)

#andere Möglichkeit zum codieren
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
#print(class_le.inverse_transform(y))

"""One Hot Encoding für nominale Merkmale"""
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

#So wird jedoch nominalen Features ein Rang gegeben --> deshalb One Hot Encoding
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray() #toarray, oder im ohe sparse=False angeben

#oder get dummies nutzen um alle Zeichenketten zu One Hot Encoden
print(pd.get_dummies(df[['price', 'color', 'size']]))
