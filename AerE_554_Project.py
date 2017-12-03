"""
Code for AerE 554 Project 1. I am attempting to fit a parabola to a cost function and extract the
material parameters (n & k, real and imaginary parts of index of refraction) from a test sample.
The cost function has been provided as pickled data which must be loaded in.
"""
import pdb
import sys
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.optimize import curve_fit, minimize

import sm_functions as sm
import half_space as hs
import util

# load in THzDataClass that I created
sys.path.insert(0, 'D:\\PycharmProjects\\THzProcClass')
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

t0 = time.time()

try:
    with open('total_cost.pickle', 'rb') as f:
        cost = pickle.load(f)
    f.close()

except FileNotFoundError:
    # build the cost function over the bounds
    for i, nr in enumerate(nr_bounds):
        # print(i)
        for j, ni in enumerate(ni_bounds):
            n = np.array([nr, ni])

            raw_cost = \
                hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                                 ref_freq[:stop_index], d, theta0)[30]

            # try to emulate least squares
            cost[i, j] = np.sum(raw_cost)

    with open('total_cost.pickle', 'wb') as f:
        pickle.dump(cost, f)

cost_min_coords = np.argmin(cost)
cost_min_coords = np.unravel_index(cost_min_coords, cost.shape)

# estimate of n from brute force search
n_hat_real = nr_bounds[cost_min_coords[0]]
n_hat_imag = ni_bounds[cost_min_coords[1]]
n_hat = n_hat_real + 1j * n_hat_imag

t1 = time.time()
print('Brute Force Search Time = %0.4f seconds' % (t1-t0))
print(n_hat)

# try optimizing the true function and see how long it takes
# determine the time of flight between the front and back surface echos
t0 = data.time[data.waveform[location[0], location[1], :].argmax()]
t1 = data.time[data.waveform[location[0], location[1], gate1:].argmax() + gate1]

# since we know the thickness of the sample, we can use that to get a first estimate of the index
# of refraction, initial estimate assumes no angle of incidence
c_sample = 2*d / (t1-t0)  # estimate of speed of light in the material
n0 = c / c_sample  # initial guess for n_real
n0 = complex(n0, 0)

total_error = 100
iter = 0
while total_error > 0.1 and iter < 1000:
    theta1 = sm.get_theta_out(1, n0, theta0)
    model = hs.half_space_model(e0[:stop_index], ref_freq[:stop_index], n0, d, theta0, theta1)

    true_phase = np.unwrap(np.angle(e2[:stop_index]))
    model_phase = np.unwrap(np.angle(model[:stop_index]))

    true_magnitude = np.abs(e2[:stop_index])
    model_magnitude = np.abs(model[:stop_index])

    phase_error = true_phase - model_phase
    magnitude_error = true_magnitude - model_magnitude

    n_new = n0.real + 0.001 * phase_error[30]
    k_new = n0.imag + 0.001 * magnitude_error[30]

    total_error = np.abs(magnitude_error[30]) + np.abs(phase_error[30])

    n0 = complex(n_new, k_new)

    iter += 1

pdb.set_trace()

# number of samples to use in each row of the sampling plan
n_samples = 6

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
                                                ref_freq[:stop_index], d, theta0)[30]

    sampled_values[i] = np.sum(raw_cost)

p0 = (-0.1, 4.167e-3, 50, 1.7, 3.8676e-3)
popt, pcov = curve_fit(util.parabolic_equation, sample_points, sampled_values, maxfev=2500)

n_guess = np.array([2.5, -1])
result = minimize(util.parabolic_equation, n_guess, args=(popt[0], popt[1], popt[2], popt[3],
                                                          popt[4]))
n_model_sol = result.x

nr_meshed, ni_meshed = np.meshgrid(nr_bounds, ni_bounds, indexing='ij')
n_meshed = np.array([nr_meshed, ni_meshed])
parabolic_fit = util.parabolic_equation(n_meshed, *popt)

best_fit_min_coords = np.unravel_index(parabolic_fit.argmin(), parabolic_fit.shape)

# estimate of n from brute force search
n_parab_real = nr_bounds[best_fit_min_coords[0]]
n_parab_imag = ni_bounds[best_fit_min_coords[1]]
n_parab = n_parab_real + 1j * n_parab_imag

t1 = time.time()

print()
print('Time to build parabolic model and minimize:', t1-t0)
print(n_parab)

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
plt.grid()

# currently axis labels on mlab figure don't seem to be working
# looks to be a bug on their end
mlab.figure('Cost Function vs n')
mlab.surf(ni_bounds, nr_bounds, np.rot90(cost, -1), warp_scale='auto')
mlab.colorbar()

plt.figure('Parabolic Approximation of Cost Function')
im = plt.imshow(parabolic_fit, aspect='auto', interpolation='none', extent=extent)
plt.scatter(x=n_parab.imag, y=n_parab.real, color='r', label='Minimum')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.colorbar(im)
plt.grid()

plt.figure('Typical Waveform')
plt.plot(data.time, data.waveform[location[0], location[1], :], 'r')
plt.axvline(data.time[gate1], color='k', linestyle='--')
plt.title('Typical Waveform')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.show()

