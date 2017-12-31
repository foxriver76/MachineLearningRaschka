#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 12:13:57 2017

@author: moritz
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-' \
                 'databases/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1)
    
"""Logistic Regression Pipeline"""
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(penalty='l2',
                                               random_state=0,
                                               C=100.0))])

X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3,
                          random_state=1).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             lw=1,
             label='ROC-Teilmenge %d (Fläche = %0.2f)'
             % (i+1, roc_auc))

plt.plot([0,1],
         [0, 1],
         linestyle='--',
         color = (0.6, 0.6, 0.6),
         label='Zufälliges Raten')
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mittelwert ROC (Fläche = %0.2f)'
         % mean_auc, lw=2)

plt.plot([0, 0, 1],
         [0, 1, 1],
         lw=2,
         linestyle=':',
         color='black',
         label='Perfektes Ergebnis')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Falsch-Positiv-Rate')
plt.ylabel('Richtig-Positiv-Rate')
plt.title('Receiver Operator Chararcteristic')
plt.legend(loc='lower right')
plt.show()

"""Wenn nur ROC Fläche interessant, dann:"""
pipe_lr = pipe_lr.fit(X_train2, y_train)
y_pred2 = pipe_lr.predict(X_test[:, [4, 14]])
print('ROC AUC: %.3f'
      % roc_auc_score(y_true=y_test, y_score=y_pred2))
print('Korrektklassifizierungsrate: %.3f'
      % accuracy_score(y_true=y_test, y_pred=y_pred2))