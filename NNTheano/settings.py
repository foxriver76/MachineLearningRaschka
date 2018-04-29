#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:02:30 2018

@author: moritz
"""
import theano
from theano import tensor as T


#Initialisieren
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

#Kompilieren
net_input = theano.function(inputs=[w1, x1, w0],
                            outputs=z1)

#Ausführen
print('Nettoeingabe: %2.f' % net_input(2.0, 1.0, 0.5))

"""Standardeinstellungen für Gleitkommazahlen"""
print(theano.config.floatX)

"""Ändern auf float32"""
#theano.config.floatX = 'float32'

"""In der Kommandozeile muss folgendes ausgeführt werden um auf GPUs aktuell 
zu arbeiten: export THEANO_FLAGS=floatX=float32"""

"""Standardeinstellung wo Code ausgeführt wird"""
print(theano.config.device)

"""Ohne Änderungen am Code können wir durch Ausführung aus der Shell
wählen weie wir ausführen möchten:
    CPU: THEANO_FLAGS=device=cpu,floatX=float64 python skript.py
    GPU: THEANO_FLAGS=device=gpu,floatX=float32 python skript.py
"""

"""Man kann Einstellungen auch permanent speichern indem man eine Datei namens 
.theanorc im Home-Verzeichnis erstellt mit beispielsweise folgendem Inhalt
    [global]
    floatX=float32
    device=gpu
"""