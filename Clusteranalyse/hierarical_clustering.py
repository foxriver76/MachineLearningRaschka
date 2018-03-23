#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:06:55 2018

@author: moritz
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
import matplotlib.pyplot as plt


"""Random Daten erzeugen"""
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

"""Distanzmatrix erzeugen"""
row_dist = pd.DataFrame(squareform(
        pdist(df, metric='euclidean')),
        columns=labels, index=labels)
print(row_dist)

"""Complete Linkage Agglomerative Clustering"""
# help(linkage)

"""Ansatz obere Dreiecksmatrix"""
row_clusters = linkage(pdist(df, metric='euclidean'),
                       method='complete')

# oder 
row_clusters = linkage(df.values,
                       method='complete',
                       metric='euclidean')

"""Kopplungsmatrix ausgeben"""
print(pd.DataFrame(row_clusters,
                   columns=['Zeile 1',
                            'Zeile 2',
                            'Distanz',
                            '# Objekte im Cluster'],
                    index=['Cluster %d' %(i+1) for i in 
                           range(row_clusters.shape[0])]))

"""Dendrogramm visualisieren"""
set_link_color_palette(['black']) # sonst Dendrogramm bunt
row_dendr = dendrogram(row_clusters,
                       labels=labels,
                       color_threshold=np.inf) # sonst Dendrogramm bunt
plt.tight_layout()
plt.ylabel('Euklidische Distanz')
plt.show()
print('ID_0 und ID_4 sowie ID_1 und ID_2 sind sich am ähnlichsten')

"""Dendrogramm mit Heatmap verknüpfen"""
fig = plt.figure(figsize=(8,8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

"""Im DF umsortieren nach Cluster Labels"""
df_rowclust = df.ix[row_dendr['leaves'][::-1]]

"""Heatmap erzeugen und rechts neben Dendrogramm platzieren"""
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
                   interpolation='nearest', cmap='hot_r')

"""Aussehen und Zuordnung der Heatmap"""
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()