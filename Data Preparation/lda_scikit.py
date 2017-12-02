#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 08:28:41 2017

@author: moritz
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from Classifier.plot_decision_regions import plot_decision_regions


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)    
X_test_std = sc.transform(X_test)

"""LDA"""
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

"""Überprüfen einer Logistic Regression"""
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower left')
plt.title('Trainingsdatenmenge')
plt.show()

"""Testdatenmenge plotten"""
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower left')
plt.title('Testdatenmenge')
plt.show()