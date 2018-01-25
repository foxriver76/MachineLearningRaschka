#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:52:40 2018

@author: moritz
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

df = pd.read_csv('./movie_data.csv')

"""mit CountVectorizer Array mit Textdaten einlesen
Standardmäßig nutzt der CountVectorizer Monogramme (1-Gramme) also 
1 Wort wird gecountet. Man kann dies z.B. auf Bigramme ändern durch
ngram_range(2,2)"""
count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and \
        one is two'])
bag = count.fit_transform(docs)

"""Vokabular ausgeben - alle Worte mit Index"""
print(count.vocabulary_)

"""BoW Counter ausgeben Index ist von Vokabular"""
print(bag.toarray())

"""Wortrelevanz mit tf-idf Maß beurteilen"""
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(
        count.fit_transform(docs)).toarray())