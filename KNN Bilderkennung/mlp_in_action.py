#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:49:19 2018

@author: moritz
"""

from neuronales_netz import NeuralNetMLP
from readData import load_mnist
import matplotlib.pyplot as plt
import pickle
import os


"""classifier laden"""
nn = pickle.load(open(os.path.join('NeuralNetSaved/pkl_objects', 'neural_net.pkl'), 'rb'))

"""Daten einlesen"""
X_train, y_train = load_mnist('Data', kind='train')

X_test, y_test = load_mnist('Data', kind='t10k')

"""784-50-10-MLP initialisieren (784 Eingabeeinheiten (n_features) +  50 verdeckte
Einheiten (n_hidden) + 10 Ausgabeeinheiten (n_outputs)))"""
"""
nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=1000,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  shuffle=True,
                  minibatches=50,
                  random_state=1)

nn.fit(X_train, y_train, print_progress=True)
"""

"""Jede 50. Schritt plotten da 50 Teilmengen mit je 1000 Epochen (Wdh)"""
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Wert der Straffunktion')
plt.xlabel('Epochen * 50')
plt.tight_layout()
plt.show()

"""Modell als Byte Code speichern/serialisieren"""
"""
dest = os.path.join('NeuralNetSaved', 'pkl_objects')

if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(nn,
            open(os.path.join(dest, 'neural_net.pkl'), 'wb'),
            protocol=4)
"""