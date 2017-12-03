"""
Test code to see what initial guess is needed for paraboloid
"""

import numpy as np
import matplotlib.pyplot as plt

import util

x = np.linspace(-0.001, -2, 250)
y = np.linspace(4.25, 1, 250)

# initial guess for the paraboloid parameters
p0 = (-0.1, 4.167e-3, 50, +1.7, 3.8676e-3)

# yy, xx = np.meshgrid(y, x, indexing='ij')
# n_meshed = np.array([x, y])

paraboloid = np.zeros((len(y), len(x)))
for i in range(len(y)):
    for j in range(len(x)):
        data = (y[i], x[j])
        paraboloid[i, j] = util.parabolic_equation(data, *p0)

extent = (x[1], x[-1], y[-1], y[1])

plt.figure('Test of Parabolic Equation')
plt.imshow(paraboloid, interpolation='none', aspect='auto', extent=extent)
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.colorbar()
plt.grid()
