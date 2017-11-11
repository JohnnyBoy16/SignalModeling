"""
Attempting to build a model that will replicate the paraboloid results (figure 2) in Duvillaret's
1996 paper on material parameter extraction with THz spectroscopy
"""
import pdb
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

import sm_functions as sm
import half_space as hs

plt.ioff()  # turn off interactive plotting

ref_file = 'C:\\Work\\Signal Modeling\\References\\ref 18OCT2017\\30ps waveform.txt'

nr = np.array([1.0, 1.75, 1.0])  # real part
ni = np.array([0.0, -0.05, 0.0])  # imaginary part (attenuation)

nr_guess = np.linspace(1, 4.5, 250)
ni_guess = np.linspace(-0.001, -0.2, 250)

n_layers = 1  # the number of layers in the structure
d = np.array([0.5])  # thickness of the layers in mm

c = 0.2998  # speed of light in mm / ps

gate1 = 1613  # gates to slice out the back surface echo
gate2 = 2802

max_f = 2.5

# index to remove front "blip"
slice_loc = 450

# incoming angle of system is 17.5 degrees
# convert to radians
theta0 = 17.5 * np.pi / 180

n = nr + 1j*ni

ref_time, ref_amp = sm.read_reference_data(ref_file)

# adjust ref_time so initial value is 0 ps
ref_time -= ref_time[0]

# remove front blip and add ramp up factor to help with FFT
ref_amp[:slice_loc] = 0
ref_amp[slice_loc] = ref_amp[slice_loc+1] / 2

# determine the wave length that was gathered
# calculate dt and df
wave_length = len(ref_time)
dt = ref_time[1]
df = 1 / (wave_length * dt)

ref_freq = np.linspace(0., (wave_length/2) * df, wave_length//2+1)

ref_freq_amp = np.fft.rfft(ref_amp) * dt
e0 = -ref_freq_amp  # initial E is the opposite of reference

# calculate the frequency index that is closest to max_f
stop_index = np.argmin(np.abs(ref_freq - max_f))
step = stop_index

lb = step // 2  # lower bound
ub = step // 2 + 1  # upper bound

# begin the dummy signal model
# calculate angle in each layer
theta = np.zeros(n_layers+2, dtype=complex)
theta[0] = theta0
theta[-1] = theta0
for i in range(1, n_layers+1):
    theta[i] = sm.get_theta_out(n[i-1], n[i], theta[i-1])

gamma = sm.global_reflection_model(n, theta, ref_freq, d, n_layers)

e1 = e0 * gamma[0]

return_wave = np.fft.irfft(e1) / dt

bse = np.zeros(wave_length)
bse[gate1:gate2] = return_wave[gate1:gate2]
bse[gate1-1] = bse[gate1] / 2
bse[gate2] = bse[gate2-1] / 2

e2 = np.fft.rfft(bse) * dt

# the cost map over given nr & ni list
cost = np.zeros((len(nr_guess), len(ni_guess), stop_index))
error = np.zeros((len(nr_guess), len(ni_guess), stop_index), dtype=complex)
arg_T = np.zeros(cost.shape)
log_abs_T = np.zeros(cost.shape)

for i, nr in enumerate(nr_guess):
    print(i)
    for j, ni in enumerate(ni_guess):
        n = np.array([nr, ni])

        raw_cost, raw_error, model = \
            hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                             ref_freq[:stop_index], d, theta0)

        # try to emulate least squares
        cost[i, j, :] = np.sum(raw_cost)

        # Dr. Chiou wants to check the error at each (nr, ni) pair
        error[i, j, :] = np.sum(raw_error)

        # check the unwrapped phase and log(abs(T)) like in Duvillaret's paper
        arg_T[i, j, :] = np.unwrap(np.angle(model))
        log_abs_T[i, j, :] = np.log(np.abs(model))

# dump the pickled cost data so it can be used for AerE 554 optimization project
with open('cost.pickle', 'wb') as f:
    pickle.dump(cost[:, :, 31], f)
f.close()

# at the moment we are solving for a single value of n, so it doesn't matter what the
# last index value is
min_coords = np.unravel_index(np.argmin(cost[:, :, 31]), cost[:, :, 0].shape)

# build a new model based on argmin of brute force search results
# use the half space model here
nr_hat = nr_guess[min_coords[0]]
ni_hat = ni_guess[min_coords[1]]
n_hat = nr_hat + 1j*ni_hat

theta1_hat = sm.get_theta_out(1.0, n_hat, theta0)

# front surface reflection coefficient
r01 = sm.reflection_coefficient(1.0, n_hat, theta0, theta1_hat)

# Fresnel reflection coefficients for the BSE
t01 = sm.transmission_coefficient(1.0, n_hat, theta0, theta1_hat)
r10 = sm.reflection_coefficient(n_hat, 1.0, theta1_hat, theta0)
t10 = sm.transmission_coefficient(n_hat, 1.0, theta1_hat, theta0)

FSE = e0 * r01

# create back surface echo and shift appropriately
time_delay = 2*n_hat*d / (c*np.cos(theta1_hat))
shift = np.exp(-1j * 2*np.pi * ref_freq * time_delay)
BSE = e0 * t01 * r10 * t10 * shift

e1_hat = FSE + BSE  # create waveform in frequency domain

return_wave_hat = np.fft.irfft(e1_hat) / dt  # convert to time domain

plt.figure('Reference Waveform')
plt.plot(ref_time, ref_amp, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('E0 Frequency Waveform')
plt.plot(ref_freq, np.abs(e0), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3)
plt.grid()

plt.figure('Dummy Sample Signal')
plt.plot(ref_time, return_wave, 'r')
plt.axvline(ref_time[gate1], color='k', linestyle='--')
plt.axvline(ref_time[gate2], color='k', linestyle='--')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Dummy Sample BSE only')
plt.plot(ref_time, bse, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('BSE in Frequency Domain')
plt.plot(ref_freq, np.abs(e2), 'r')
plt.xlim(0, 3)
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()

extent = (ni_guess[0], ni_guess[-1], nr_guess[-1], nr_guess[0])

plt.figure('Cost vs. Index of Refraction')
cost_map = plt.imshow(cost[:, :, 31], interpolation='none', extent=extent, aspect='auto')
plt.scatter(x=ni_guess[min_coords[1]], y=nr_guess[min_coords[0]], color='r',
            label='Min Value Location')
plt.title('Cost vs. Index of Refraction')
plt.xlabel(r'$n_{imag}$')
plt.ylabel(r'$n_{real}$')
plt.colorbar(cost_map)
plt.grid()
plt.legend()

plt.figure('Return Wave from n Estimate')
plt.plot(ref_time, return_wave_hat, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

# for i in range(stop_index):
#     fig_string = 'Arg(T) vs n @ freq = %0.2f THz' % ref_freq[i]
#     plt.figure(fig_string)
#     plt.imshow(arg_T[:, :, i], interpolation='none', extent=extent, aspect='auto')
#     plt.title(fig_string)
#     plt.xlabel(r'$n_{imag}$')
#     plt.ylabel(r'$n_{real}$')
#     plt.grid()
#     plt.colorbar()
#     plt.savefig('image_bin\\' + fig_string + '.png')
#     if i != stop_index-1:
#         plt.close()
#
#     fig_string = 'Log Abs(T) vs n @ freq = %0.2f THz' % ref_freq[i]
#     plt.figure(fig_string)
#     plt.imshow(log_abs_T[:, :, i], interpolation='none', extent=extent, aspect='auto')
#     plt.title('Log Abs(T) vs n')
#     plt.xlabel(r'$n_{imag}$')
#     plt.ylabel(r'$n_{real}$')
#     plt.grid()
#     plt.colorbar()
#     plt.savefig('image_bin\\' + fig_string + '.png')
#     if i != stop_index-1:
#         plt.close()

mlab.figure('Arg(T) vs n')
mlab.surf(arg_T[:, :, 31])
mlab.colorbar()

mlab.figure('Log Abs(T) vs n')
mlab.surf(log_abs_T[:, :, 31])
mlab.colorbar()

plt.figure('Error between Real Components')
plt.imshow(error[:, :, 31].real, interpolation='none', extent=extent, aspect='auto')
plt.title('Error between Real Components')
plt.xlabel(r'$n_{imag}$')
plt.ylabel(r'$n_{real}$')
plt.grid()
plt.colorbar()

plt.figure('Error between Imaginary Components')
plt.imshow(error[:, :, 31].imag, interpolation='none', extent=extent, aspect='auto')
plt.title('Error between Imaginary Components')
plt.xlabel(r'$n_{imag}$')
plt.ylabel(r'$n_{real}$')
plt.grid()
plt.colorbar()

min_coords_real = np.argmin(np.abs(error[:, :, 31].real))
min_coords_real = np.unravel_index(min_coords_real, error[:, :, 0].shape)

min_coords_imag = np.argmin(np.abs(error[:, :, 31].imag))
min_coords_imag = np.unravel_index(min_coords_imag, error[:, :, 0].shape)

plt.figure('Absolute Real Error')
im = plt.imshow(np.abs(error[:, :, 31].real), interpolation='none', extent=extent, aspect='auto')
# plt.scatter(x=ni_guess[min_coords_real[1]], y=nr_guess[min_coords_imag[0]], color='r')
plt.title('Absolute Real Error')
plt.xlabel(r'$n_{imag}$')
plt.ylabel(r'$n_{real}$')
plt.grid()
plt.colorbar(im)

plt.figure('Absolute Imaginary Error')
im = plt.imshow(np.abs(error[:, :, 31].imag), interpolation='none', extent=extent, aspect='auto')
# plt.scatter(x=ni_guess[min_coords_imag[1]], y=nr_guess[min_coords_real[0]], color='r')
plt.title('Absolute Imaginary Error')
plt.xlabel(r'$n_{imag}$')
plt.ylabel(r'$n_{real}$')
plt.grid()
plt.colorbar(im)

plt.ion()

plt.show()
