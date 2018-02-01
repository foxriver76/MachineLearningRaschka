#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:16:42 2018

@author: moritz
"""

import pickle
import os


"""Modell als Byte Code speichern/serialisieren
clf und stop kommen auf Sentiment Analysis -> out_of_core_learning"""

dest = os.path.join('movieclassifier', 'pkl_objects')

if not os.path.exists(dest):
    os.makedirs(dest)
    
pickle.dump(stop,
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol=4)
pickle.dump(clf,
            open(os.path.join(dest, 'classifier.pkl'), 'wb'),
            protocol=4)