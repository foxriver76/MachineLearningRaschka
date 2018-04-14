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
import numpy as np


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

"""Erneut plotten mit Durchschnittswert pro Minibatch plotten"""
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(len(cost_avgs)),
         cost_avgs,
         color='red')
plt.ylim([0, 2000])
plt.ylabel('Wert der Straffunktion')
plt.xlabel('Epochen')
plt.tight_layout()
plt.show()

"""Güte durch Korrektklassifizierungsrate aufzeigen"""
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Korrektklassifizierungsrate Training: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Korrektklassifizierungsrate Test %.2f%%' % (acc * 100))

"""Zahlen ansehen, die dem NN Probleme machen"""
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'
      % (i+1, correct_lab[i], miscl_lab[i]))
ax[0].set_yticks([])
ax[0].set_xticks([])
plt.tight_layout()
plt.show()