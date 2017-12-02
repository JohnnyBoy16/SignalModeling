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
from scipy.optimize import curve_fit

import sm_functions as sm
import half_space as hs
import util

# load in THzDataClass that I created
sys.path.insert(0, 'D:\\Python\\THzProcClass')
from THzData import THzData


ref_file = 'D:\\Work\\Signal Modeling\\References\\ref 18OCT2017\\30ps waveform.txt'
tvl_file = 'D:\\Work\\Signal Modeling\\THz Data\\Shim Stock\\New Scans\\Yellow Shim Stock.tvl'

# range of real and imaginary values to build the cost function over
nr_bounds = np.linspace(4.25, 1, 250)
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

        raw_cost = \
            hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                             ref_freq[:stop_index], d, theta0)

        # try to emulate least squares
        cost[i, j] = np.sum(raw_cost)

cost_min_coords = np.argmin(cost)
cost_min_coords = np.unravel_index(cost_min_coords, cost.shape)

# estimate of n from brute force search
n_hat_real = nr_bounds[cost_min_coords[0]]
n_hat_imag = ni_bounds[cost_min_coords[1]]
n_hat = n_hat_real + 1j * n_hat_imag

# number of samples to use in the latin hypercube
n_samples = 4

t0 = time.time()

# build a latin hypercube sampling plan for use in RBF and Kriging metamodels
# user criterion='maximin' to maximize the minimum distance between samples and get better
# coverage of design space
nr_samples = np.linspace(nr_bounds[0], nr_bounds[-1], n_samples)
ni_samples = np.linspace(ni_bounds[0], ni_bounds[-1], n_samples)

sample_points = np.zeros((2, n_samples**2))
k = 0
for i in range(n_samples):
    for j in range(n_samples):
        sample_points[0, k] = nr_samples[i]
        sample_points[1, k] = ni_samples[j]
        k += 1

# calculate the value of the function at the sample points
sampled_values = np.zeros(n_samples**2)
for i in range(n_samples**2):
    n = sample_points[:, i]

    raw_cost = hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                                ref_freq[:stop_index], d, theta0)

    sampled_values[i] = np.sum(raw_cost)

popt, pcov = curve_fit(util.parabolic_equation, sample_points, sampled_values)

nr_meshed, ni_meshed = np.meshgrid(nr_bounds, ni_bounds, indexing='ij')
n_meshed = np.array([nr_meshed, ni_meshed])
parabolic_fit = util.parabolic_equation(n_meshed, *popt)

t1 = time.time()
print('Total Time:', t1-t0)

# use extent to set the values on the axis label for plotting
extent = (ni_bounds[0], ni_bounds[-1], nr_bounds[-1], nr_bounds[0])

plt.figure('Cost Function vs n')
im = plt.imshow(cost, aspect='auto', interpolation='none', extent=extent)
plt.scatter(x=n_hat_imag, y=n_hat_real, color='r', label='Min Value Location')
plt.scatter(x=sample_points[1, :], y=sample_points[0, :], color='k', label='Sample Points')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.colorbar(im)
plt.legend()
# plt.xlim(0, ni_bounds[-1])
plt.grid()

# currently axis labels on mlab figure don't seem to be working
# looks to be a bug on their end
mlab.figure('Cost Function vs n')
mlab.surf(ni_bounds, nr_bounds, np.rot90(cost, -1), warp_scale='auto')
mlab.colorbar()

plt.figure('Parabolic Approximation of Cost Function')
plt.imshow(parabolic_fit, aspect='auto', interpolation='none', extent=extent)
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.colorbar()
plt.grid()

plt.show()

