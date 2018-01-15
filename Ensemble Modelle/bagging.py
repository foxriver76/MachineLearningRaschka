#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:38:18 2018

@author: moritz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Klassenbezeichnung', 'Alkohol', 
                   'Apfelsäure', 'Asche', 
                   'Aschaelkalität', 'Magnesium', 
                   'Phenole insgesamt', 'Flavanoide',
                   'Nicht flavanoide Phenole',
                   'Tannin',
                   'Farbintensität', 'Farbe',
                   'OD280/OD315 des verdünnten Weins',
                   'Prolin']
df_wine = df_wine[df_wine['Klassenbezeichnung'] != 1]
y = df_wine['Klassenbezeichnung'].values
X = df_wine[['Alkohol', 'Farbe']].values

"""Klassnbezeichnung binär machen und Train & Test splitten"""
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test =\
            train_test_split(X, y, 
                             test_size=0.40,
                             random_state=1)
            
"""Bagging Classifier nutzen"""
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=None,
                              random_state=1)
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)

"""Korrektklassifizierungsrate berechnen"""
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Entscheidungsbaum Training/Test-KKR %.3f/%.3f'
      % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging Training/Test KKR %.3f/%.3f'
      %(bag_train, bag_test))

"""Entscheidungsbreiche von Bagging und Decision Tree Classifier vergleichen"""
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(8,3))
for idx, clf, tt in zip([0, 1],
                        [tree, bag],
                        ['Entscheidungsbaum', 'Bagging']):
    clf.fit(X_train, y_train)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
         X_train[y_train==0, 1],
         c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
         X_train[y_train==1, 1],
         c='red', marker='o')
    axarr[idx].set_title(tt)
    
axarr[0].set_ylabel('Alkohol', fontsize=12)
plt.text(10.2, -1.2,
         s='Farbe',
         ha='center', va='center', fontsize=12)
plt.show()
