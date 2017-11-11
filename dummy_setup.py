"""
Script written to compare Dr. Chiou's Rlayer model with my own
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import sys
import sm_functions as sm

# include this directory in the system path so we can import THzData
sys.path.insert(0, 'C:\\Work\\Signal Modeling')
sys.path.insert(0, 'C:\\Work\\THzProc - Vanilla')

from Rlayer_all import Rlayer
from THzProc1_12 import ReFFT2, IReFFT2

nr = np.array([1.0, 1.52, 1.0])  # real part
ni = np.array([0.0, 0.015, 0.0])  # imaginary part (attenuation)
n_layers = 1  # the number of layers in the structure
d = np.array([0.1016])  # thickness of the layers in mm

slice_loc = 400  # index which to remove front blip

theta0 = 17.5 * np.pi / 180  # the incoming angle of the THz scanner in radians

c = 0.2998  # speed of light in mm/ps

n = nr - 1j * ni  # make the complex n

ref_file = 'C:\\Work\\Signal Modelling\\References\\ref 12JUN2017\\60 ps waveform.txt'
sample_file = 'C:\\Work\\Signal Modelling\\THz Data\\HDPE Lens\\60 ps waveform.txt'
ref_time, ref_amp = sm.read_reference_data(ref_file)
time, sample_amp = sm.read_reference_data(sample_file)

ref_time -= ref_time[0]  # shift time so initial value is zero
ref_amp[:slice_loc] = 0
ref_amp[slice_loc-1] = ref_amp[slice_loc] / 2  # add ramp for FFT
sample_amp[:slice_loc] = 0
sample_amp[slice_loc-1] = sample_amp[slice_loc] / 2

dt = ref_time[-1] / (len(ref_time) - 1)
df = 1 / (len(ref_time) * dt)
freq = np.linspace(0, len(ref_time)/2*df, len(ref_time)//22)

# calculate theta in each angle
theta = np.zeros(n_layers+2, dtype=complex)
theta[0] = theta0  # first theta is just the angle of the THz scanner
theta[-1] = theta0  # last theta is also the angle of the THz scanner
for i in range(1, n_layers+1):  # the only unknowns are the angles in the material
    theta[i] = sm.get_theta_out(n[i-1], n[i], theta[i-1])

gamma = sm.global_reflection_model(n, theta, freq, d, n_layers)

thick = np.zeros(n_layers)
if type(d) is int:
    thick[:] = d
else:
    thick = d

rlayer_gamma = np.zeros(len(freq), dtype=complex)
for i in range(len(freq)):
    rlayer_gamma[i] = Rlayer(n_layers, n, thick, theta0, freq[i])[0]

ref_freq = ReFFT2(ref_amp, dt)
sample_freq = ReFFT2(sample_amp, dt)

# initial wave is the opposite of reference since reflection coefficient for aluminum if -1
e0 = -ref_freq

e_return = e0 * gamma[0]
e_return2 = e0 * rlayer_gamma

return_wave = IReFFT2(e_return, dt)
return_wave2 = IReFFT2(e_return2, dt)

plt.figure('Trial Waveform')
# plt.plot(ref_time, ref_amp, 'b', label='Reference')
plt.plot(ref_time, return_wave, 'r', label='Simulation')
plt.plot(ref_time, sample_amp, 'b', label='PE Window')
plt.axvline(ref_time[slice_loc], color='g', linestyle='--')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude (a.u.)')
plt.legend()
plt.grid()

plt.figure('Simulation')
plt.plot(ref_time, return_wave, 'r', label='My Simulation')
plt.plot(ref_time, return_wave2, 'b', label="Dr. Chiou's Model")
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude (a.u.)')
plt.legend()
plt.grid()

plt.figure('Frequency Waveform Comparison')
plt.plot(freq, np.abs(ref_freq), 'r', label='Reference')
plt.plot(freq, np.abs(e_return), 'b', label='John')
plt.plot(freq, np.abs(sample_freq), 'g', label='PE Window')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 4)
plt.legend()
plt.grid()

plt.figure('Real Part of Frequency')
plt.plot(freq, ref_freq.real, 'r', label='Reference')
plt.plot(freq, e_return.real, 'b', label='John')
plt.plot(freq, sample_freq.real, 'g', label='PE Window')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 4)
plt.legend()
plt.grid()

plt.figure('Imaginary Part of Frequency')
plt.plot(freq, ref_freq.imag, 'r', label='Reference')
plt.plot(freq, e_return.imag, 'b', label='John')
plt.plot(freq, sample_freq.imag, 'g', label='PE Window')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 4)
plt.legend()
plt.grid()

plt.figure('Reflection Coefficient Comparison')
plt.plot(freq, np.abs(gamma[0, :]), 'r', label='John')
plt.plot(freq, np.abs(rlayer_gamma), 'b', label='Dr. Chiou')
plt.xlabel('Frequency (THz)')
plt.ylabel('Reflection Coefficient')
plt.xlim(0, 4)
plt.legend()
plt.grid()

plt.show()
