# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z=np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k') # vertical line by x = 0 
plt.ylim(-0.1, 1.1) # whole chart y-height
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0]) # Beschriftung y Achse
ax = plt.gca()  # get current axes
ax.yaxis.grid(True) # make grid on the current axes 

plt.show()
