#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:29:31 2017

@author: moritz
"""

import numpy as np
import matplotlib.pyplot as plt

def gini(p):
    return (p)*(1- (p)) + (1 - p)*(1- (1 - p))

def entropy(p):
    return - p*np.log2(p) - (1-p)*np.log2((1-p))

def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                           ['Entropie', 'Entropie (skaliert)', 
                            'Gini-Koeffizient',
                            'Klassifizierungsfehler'],
                            ['-', '-', '--', '-.'],
                            ['black', 'lightgray',
                             'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, 
                   linestyle=ls, lw=2, color=c)
    
ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()