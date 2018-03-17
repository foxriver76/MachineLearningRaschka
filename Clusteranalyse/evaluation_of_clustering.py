#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:05:36 2018

@author: moritz
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

"""Create Sample Data"""
X, y = make_blobs(n_samples=150,
                  centers=3,
                  n_features=2,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

"""k_means auf die Daten anwenden"""
km = KMeans(n_clusters=3, 
             init='random',
             n_init=10,
             max_iter=300,
             tol=1e-04,
             random_state=0)

y_km = km.fit_predict(X)


"""Summe quadrierter Abweichungen innerhalb des Clusters
zur Evaluation -> Verzerrung genannt"""
print('Verzerrung: % .2f' % km.inertia_)

"""Ellenbogenkriterium --> Plotten der Verzerrung für unterschiedliche 
k-Werte"""
distortions = []
for i in range(1, 11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Summe quadrierter Abweichungen - Verzerrung')
plt.title('Elebogenkriterium k=3 ist ein guter Wert')
plt.show()

"""Silhouettenkoeffizient durchschnittliche Entfernung zu eigenen Punkten im
Verhältnis zu durchschnittlichen Entfernung zu Punkten anderer Cluster"""
y_ax_lower, y_ax_upper = 0, 0
yticks = []
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color='red',
            linestyle='--')
plt.title('Guter Wert für Silhouettenkoeffizient')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouettenkoeffizient')
plt.show() 

"""Vergleichsweise schlechtes Clustering k=2"""
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0],
            X[y_km==0,1],
            s=50, c='lightgreen',
            marker='s',
            label='Cluster 1')

plt.scatter(X[y_km==1,0],
            X[y_km==1,1],
            s=50, c='orange',
            marker='o',
            label='Cluster 2')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='Zentroiden')

plt.legend()
plt.grid()
plt.show()

"""Silhouettendiagramm erstellen"""
