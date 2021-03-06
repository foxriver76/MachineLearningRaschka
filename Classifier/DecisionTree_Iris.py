#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:18:20 2017

@author: moritz
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plot_decision_regions import plot_decision_regions
from sklearn.tree import export_graphviz


"""import iris dataset and splitting data"""
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0) #30% Testdaten, 70% Trainingsdaten

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))
plt.xlabel('Länge des Blütenblattes [cm]')
plt.ylabel('Breite des Blütenblattes [cm]')
plt.legend(loc='upper left')
plt.show()


"""export decision tree for graphViz"""
export_graphviz(tree,
                out_file='tree.dot',
                feature_names=['Länge des Blütenblattes',
                               'Breite des Blütenblattes'])