#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:22:47 2017

@author: moritz
"""

import pandas as pd

"""Dataframe mit nominalen (color), ordinalem (Größe) und numersichen (price) Feature"""
df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

"""Mappen der ordinalen Features in Nummerische"""
size_mapping = {
        'XL':3,
        'L':2,
        'M':1}

df['size'] = df['size'].map(size_mapping)

inv_size_mapping = {v: k for k, v in size_mapping.items()}