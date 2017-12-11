#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:49:07 2017

@author: moritz
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel(X, gamma, n_components):
    """Implementierung der RBF-Kernel-PCA
    
    Parameter
    ---------
    X: {numpy ndarray}, shape = [n_samples, n_features]
    
    gamma: float
        Parameter zum abstimmen des RBF-Kernels
        
    n_components: int
        Anzahl der zurückgelieferten Hauptkomponenten
        
    Rückgabewert
    ------------
    X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
        Projizierte Datenmenge
    """

    # Paarweise quadratische euklidische Abstände in der
    # NxN-dimensionalen Datenmenge berechnen
    sq_dists = pdist(X, 'sqeuclidean')
    
    # Abstände paarweise in qudaratische Matrix umwandeln
    mat_sq_dists = squareform(sq_dists)
    
    # Symmetrische Kernel-Matrix berechnen
    K = exp(-gamma * mat_sq_dists)
    
    # Kernel -Matrix zentrieren
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) \
                        + one_n.dot(K).dot(one_n)
                    
    # Paare aus Eigenwerten und Eigenvektoren der 
    # zentrierten Kernel-Matrix ermitteln; numpy.eigh gibt
    # diese sortiert zurück
    eigvals, eigvecs = eigh(K)
    
    # Die k wichtigsten Eigenvektoren ermitteln
    X_pc = np.column_stack((eigvecs[:, -i]
        for i in range(1, n_components + 1)))
    
    return X_pc


def rbf_kernel_pca(X, gamma, n_components):
    """Implementierung der RBF-Kernel-PCA
    die auch Eigenwerte der Kernelmatrix zurückgibt
    
    Parameter
    ---------
    X: {numpy ndarray}, shape = [n_samples, n_features]
    
    gamma: float
        Parameter zum abstimmen des RBF-Kernels
        
    n_components: int
        Anzahl der zurückgelieferten Hauptkomponenten
        
    Rückgabewert
    ------------
    X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
        Projizierte Datenmenge
    """

    # Paarweise quadratische euklidische Abstände in der
    # NxN-dimensionalen Datenmenge berechnen
    sq_dists = pdist(X, 'sqeuclidean')
    
    # Abstände paarweise in qudaratische Matrix umwandeln
    mat_sq_dists = squareform(sq_dists)
    
    # Symmetrische Kernel-Matrix berechnen
    K = exp(-gamma * mat_sq_dists)
    
    # Kernel -Matrix zentrieren
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) \
                        + one_n.dot(K).dot(one_n)
                    
    # Paare aus Eigenwerten und Eigenvektoren der 
    # zentrierten Kernel-Matrix ermitteln; numpy.eigh gibt
    # diese sortiert zurück
    eigvals, eigvecs = eigh(K)
    
    # Die k wichtigsten Eigenvektoren ermitteln
    alphas = np.column_stack((eigvecs[:, -i]
        for i in range(1, n_components + 1)))
    
    # Korrespondierende Eigenwerte ermitteln
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]
    
    return alphas, lambdas
    
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum(
            (x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    
    return k.dot(alphas / lambdas)