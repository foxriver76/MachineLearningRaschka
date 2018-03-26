#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:32:01 2018

@author: moritz
"""

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

"""Sample Daten erzeugen"""
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)

plt.scatter(X[:, 0], X[:, 1])
plt.show()