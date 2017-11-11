"""
Code for AerE 554 Project 1. I am attempting to fit a parabola to a cost function and extract the
material parameters (n & k, real and imaginary parts of index of refraction) from a test sample.
The cost function has been provided as pickled data which must be loaded in.
"""
import pickle
import pdb

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from mayavi import mlab
import pyDOE

nr_bounds = [1.0, 4.5]
ni_bounds = [-0.001, -0.2]

# number of samples to use in the latin hypercube sampling plan
n_samples = 5

# first thing to do is load the cost function
with open('cost.pickle', 'rb') as f:
    cost = pickle.load(f)

# TODO build the paraboloid model

# build a latin hypercube sampling plan for use in RBF and Kriging metamodels
# user criterion='maximin' to maximize the minimum distance between samples and get better
# coverage of design space
sample_points = pyDOE.lhs(2, samples=n_samples, criterion='maximin')

# rescale the sample points to match the domain
# first column is nr values (real part of index of refraction)
# second column is ni values (imaginary part of index of refraction)
sample_points[:, 0] = sample_points[:, 0] * (nr_bounds[-1]-nr_bounds[0]) + nr_bounds[0]
sample_points[:, 1] = sample_points[:, 1] * (ni_bounds[-1]-ni_bounds[0]) + ni_bounds[0]

# build the extent values for the x and y labels on the plot. This needs to be updated whenever
# the values in validation.py are updated unfortunately
# TODO look into changing this in the future
extent = (ni_bounds[0], ni_bounds[-1], nr_bounds[-1], nr_bounds[0])

plt.figure('Cost Function vs n')
im = plt.imshow(cost, aspect='auto', interpolation='none', extent=extent)
plt.scatter(x=sample_points[:, 1], y=sample_points[:, 0], color='r', label='Sample Points')
plt.xlabel(r'$n_{imag}$')
plt.ylabel(r'$n_{real}$')
plt.colorbar(im)
plt.legend()
plt.grid()

mlab.figure('Cost Function vs n')
mlab.surf(np.rot90(cost, -1), warp_scale='auto')
mlab.colorbar()

plt.show()
