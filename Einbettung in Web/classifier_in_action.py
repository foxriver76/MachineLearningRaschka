#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 09:45:43 2018

@author: moritz
"""

import os
os.chdir('movieclassifier')

import pickle
import os
from vectorizer import vect
import numpy as np

"""classifier laden"""
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

"""label codieren"""
label = {0:'negative', 1:'positive'}

"""Vorhersage testen"""
example = ['I love this movie']
X = vect.transform(example)
print('Vorhersage: %s\nWahrscheinlichkeit: %.2f%%' %\
      (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))