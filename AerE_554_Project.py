"""
Code for AerE 554 Project 1. I am attempting to fit a parabola to a cost function and extract the
material parameters (n & k, real and imaginary parts of index of refraction) from a test sample.
The cost function has been provided as pickled data which must be loaded in.
"""
import pdb
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import pyDOE
import scipy.interpolate as interpolate
from sklearn.gaussian_process import GaussianProcessRegressor

import sm_functions as sm
import half_space as hs

# load in THzDataClass that I created
sys.path.insert(0, 'D:\\BoxSync\\PycharmProjects\\THzProcClass3')
from THzData import THzData

ref_file = 'D:\\Work\\Signal Modeling\\References\\ref 18OCT2017\\30ps waveform.txt'
tvl_file = 'D:\\Work\\Signal Modeling\\THz Data\\Shim Stock\\New Scans\\Yellow Shim Stock.tvl'

# range of real and imaginary values to build the cost function over
nr_bounds = np.linspace(1, 4.25, 250)
ni_bounds = np.linspace(-0.001, -2, 250)

location = np.array([16, 5])  # index from which to extract values from tvl file

d = np.array([0.508])  # thickness of the yellow shim stock in mm

c = 0.2998  # speed of light in mm / ps

gate0 = 450  # index to remove front "blip"
gate1 = 1565  # gates to slice out the back surface echo
gate2 = 2900

# maximum frequency we want to use in solution
# above this signal to noise level gets too low
max_f = 2.5

# incoming angle of system is 17.5 degrees
# convert to radians
theta0 = 17.5 * np.pi / 180

# load the reference signal and tvl scan data
ref_time, ref_amp = sm.read_reference_data(ref_file)
data = THzData(tvl_file)

# adjust ref_time so initial value is 0 ps
ref_time -= ref_time[0]

# remove front blip and add ramp up factor to help with FFT
ref_amp[:gate0] = 0
ref_amp[gate0] = ref_amp[gate0+1] / 2

# determine the wave length that was gathered
# calculate dt and df
wave_length = len(ref_time)
dt = ref_time[1]
df = 1 / (wave_length * dt)

ref_freq = np.linspace(0., (wave_length/2) * df, wave_length//2+1)

ref_freq_amp = np.fft.rfft(ref_amp) * dt
e0 = -ref_freq_amp  # initial E0 is the opposite of reference

# calculate the frequency index that is closest to max_f
stop_index = np.argmin(np.abs(ref_freq - max_f))
step = stop_index

lb = step // 2  # lower bound
ub = step // 2 + 1  # upper bound

# the cost map over given nr & ni list
cost = np.zeros((len(nr_bounds), len(ni_bounds)))
error = np.zeros(cost.shape, dtype=complex)
arg_T = np.zeros((len(nr_bounds), len(ni_bounds), stop_index))
log_abs_T = np.zeros(arg_T.shape)

# slice out the area around the back surface echo and add a ramp factor to help fft
data.gated_waveform = np.zeros(data.waveform.shape)
data.gated_waveform[:, :, gate1:gate2] = data.waveform[:, :, gate1:gate2]
data.gated_waveform[:, :, gate1-1] = data.gated_waveform[:, :, gate1-1] / 2
data.gated_waveform[:, :, gate2] = data.gated_waveform[:, :, gate2-1] / 2

# calculate frequency domain representation
data.freq_waveform = np.fft.rfft(data.gated_waveform, axis=2) * data.delta_t

# experimental data from a point on the scan
e2 = data.freq_waveform[location[0], location[1], :]

# build the cost function over the bounds
for i, nr in enumerate(nr_bounds):
    print(i)
    for j, ni in enumerate(ni_bounds):
        n = np.array([nr, ni])

        raw_cost, raw_error, model = \
            hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                             ref_freq[:stop_index], d, theta0)

        # try to emulate least squares
        cost[i, j] = np.sum(raw_cost)

        # Dr. Chiou wants to check the error at each (nr, ni) pair
        error[i, j] = np.sum(raw_error)

        # check the unwrapped phase and log(abs(T)) like in Duvillaret's paper
        arg_T[i, j, :] = np.unwrap(np.angle(model))
        log_abs_T[i, j, :] = np.log(np.abs(model))

cost_min_coords = np.argmin(cost)
cost_min_coords = np.unravel_index(cost_min_coords, cost.shape)

# estimate of n from brute force search
n_hat_real = nr_bounds[cost_min_coords[0]]
n_hat_imag = ni_bounds[cost_min_coords[1]]
n_hat = n_hat_real + 1j * n_hat_imag

# number of samples to use in the latin hypercube
n_samples = 15

t0 = time.time()

# build a latin hypercube sampling plan for use in RBF and Kriging metamodels
# user criterion='maximin' to maximize the minimum distance between samples and get better
# coverage of design space
sample_points = pyDOE.lhs(2, samples=n_samples, criterion='maximin')

# rescale the sample points to match the domain
# first column is nr values (real part of index of refraction)
# second column is ni values (imaginary part of index of refraction)
sample_points[:, 0] = sample_points[:, 0] * (nr_bounds[-1]-nr_bounds[0]) + nr_bounds[0]
sample_points[:, 1] = sample_points[:, 1] * (ni_bounds[-1]-ni_bounds[0]) + ni_bounds[0]

# calculate the value of the function at the sample points
sampled_values = np.zeros(n_samples)
for i in range(n_samples):
    n = sample_points[i]

    raw_cost = hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                                ref_freq[:stop_index], d, theta0)[0]

    sampled_values[i] = np.sum(raw_cost)

nr, ni = np.meshgrid(nr_bounds, ni_bounds, indexing='ij')

# build the RBF Model
rbf_function = interpolate.Rbf(sample_points[:, 0], sample_points[:, 1], sampled_values,
                               function='multiquadric')
rbf_prediction = rbf_function(nr, ni)

# determine the minimum value location of the RBF estimate
rbf_min_coords = np.argmin(rbf_prediction)
rbf_min_coords = np.unravel_index(rbf_min_coords, rbf_prediction.shape)

# build the Kriging Model
gp = GaussianProcessRegressor()
gp.fit(X=sample_points, y=sampled_values)
n_pairs = np.column_stack([nr.flatten(), ni.flatten()])
kriging_prediction = gp.predict(n_pairs).reshape(nr.shape)

# determine the minimum value location of the kriging estimate
kriging_min_coords = np.argmin(kriging_prediction)
kriging_min_coords = np.unravel_index(kriging_min_coords, kriging_prediction.shape)

# calculate the RMSE
rbf_mse = np.sum((rbf_prediction - cost)**2) / cost.size
rbf_rmse = np.sqrt(rbf_mse)

kriging_mse = np.sum((kriging_prediction - cost)**2) / cost.size
kriging_rmse = np.sqrt(kriging_mse)

t1 = time.time()
print('Total Time:', t1-t0)

# use extent to set the values on the axis label for plotting
extent = (ni_bounds[0], ni_bounds[-1], nr_bounds[-1], nr_bounds[0])

plt.figure('Cost Function vs n')
im = plt.imshow(cost, aspect='auto', interpolation='none', extent=extent)
plt.scatter(x=n_hat_imag, y=n_hat_real, color='r', label='Min Value Location')
plt.scatter(x=sample_points[:, 1], y=sample_points[:, 0], color='k', label='Sample Points')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.colorbar(im)
plt.legend()
plt.grid()

plt.figure('Contour Plot of Cost Function')
im = plt.contourf(cost, origin='image', extent=extent)
# plt.clabel(im, inline=1, fontsize=10, colors='w')
plt.scatter(x=n_hat_imag, y=n_hat_real, color='r', label='Min Value Location')
plt.scatter(x=sample_points[:, 1], y=sample_points[:, 0], color='k', label='Sample Points')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.colorbar(im)  # seems to be better to use colorbar than make labeled line
plt.legend()
plt.grid()

# currently axis labels on mlab figure don't seem to be working
# looks to be a bug on their end
mlab.figure('Cost Function vs n')
mlab.surf(ni_bounds, nr_bounds, np.rot90(cost, -1), warp_scale='auto')
mlab.colorbar()

plt.figure('RBF Estimate of Cost Function')
im = plt.imshow(rbf_prediction, aspect='auto', interpolation='none', extent=extent)
plt.scatter(x=ni_bounds[rbf_min_coords[1]], y=nr_bounds[rbf_min_coords[0]], color='r',
            label='Min Value Location')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.legend()
plt.colorbar(im)
plt.grid()

plt.figure('Kriging Estimate of Cost Function')
im = plt.imshow(kriging_prediction, aspect='auto', interpolation='none', extent=extent)
plt.scatter(x=ni_bounds[kriging_min_coords[1]], y=nr_bounds[kriging_min_coords[0]], color='r',
            label='Min Value Location')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.legend()
plt.colorbar(im)
plt.grid()

plt.show()

