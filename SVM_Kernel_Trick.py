#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:43:02 2017

@author: moritz
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""random Datensatz erzeugen mit random Labels für nichtlineare SVM (Kernel Trick)"""
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
            c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

"""train SVM"""
svm = SVC(kernel='rbf', random_state=0,
          gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

"""SVM with IRIS"""#
"""import iris dataset and splitting data"""
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0) #30% Testdaten, 70% Trainingsdaten

"""standardize data"""
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""train SVM with gamma=0.2"""
svm = SVC(kernel='rbf', random_state=0,
          gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std,
                      y_combined, classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('Länge des Blütenblattes [standardisiert]')
plt.ylabel('Breite des Blütenblattes [standardisiert]')
plt.title("Gamma = 0.02")
plt.legend(loc='upper left')
plt.show()

"""train SVM with gamma=100"""
svm = SVC(kernel='rbf', random_state=0,
          gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std,
                      y_combined, classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('Länge des Blütenblattes [standardisiert]')
plt.ylabel('Breite des Blütenblattes [standardisiert]')
plt.title("Gamma = 100")
plt.legend(loc='upper left')
plt.show()


