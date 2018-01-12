#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:18:27 2018

@author: moritz
"""

from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mehrheitsentscheidungsklassifizierer import MajorityVoteClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import product
import numpy as np


iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test =\
     train_test_split(X, y,
                      test_size=0.5, 
                      random_state=1)
     
"""Logitische Regression, DecisionTree, k-nearest Neighbour trainieren 
und vergleichen dann zu Ensemble kombinieren"""

clf1 = LogisticRegression(penalty='l2',
                          C=0.001,
                          random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],
                   ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                   ['clf', clf3]])
clf_labels = ['Logistische Regression',
              'Entscheidungsbaum', 'KNN']
print('10-fache Kreuzvalidierung:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
    
"""Kombination zum Mehrheitsentscheidungsklassifizierer"""

mv_clf = MajorityVoteClassifier(
        classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Mehrheitsentscheidung']
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print('Korrektklassifizierungsrate: %0.2f \
          (+/-) %0.2f [%s]' % (scores.mean(),
           scores.std(), label))
    
"""Bewertung und Abstimmung"""
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

for clf, label, clr, ls \
        in zip(all_clf, clf_labels, colors, linestyles):
        # Klassenbezeichnung der positiven Klasse ist 1
        y_pred = clf.fit(X_train,
                         y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                         y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr,
                 color=clr,
                 linestyle=ls,
                 label='%s (AUC = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('Falsch-Positiv-Rate')
plt.ylabel('Richtig-Positiv-Rate')
plt.show()

"""Plotten der Entscheidungsbereiche"""
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1 
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7,5))
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
            X_train_std[y_train==0, 1],
            c='blue',
            marker='^',
            s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
            X_train_std[y_train==1, 1],
            c='red',
            marker='o',
            s=50)
    axarr[idx[0], idx[1]].set_title(tt)
    
plt.text(-3.5, -4.5, 
         s='Breite des Kelchblattes [standardisiert]',
         ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5,
         s='Länge des Blütenblattes [standardisiert]',
         ha='center', va='center', 
         fontsize=12, rotation=90)
plt.show()
    
"""Parameter holen zum optimieren"""
print(mv_clf.get_params())

"""C der Logistic Regression optimieren sowie Rastersuche für DecisionTree"""
params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train, y_train)

"""Optimale Parameter ausgeben"""
cv_keys = ('mean_test_score', 'std_test_score', 'params')
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))
    
print('Beste Parameter %s' % grid.best_params_)
print('Korrektklassifizierungsrate: %.2f' % grid.best_score_)
        