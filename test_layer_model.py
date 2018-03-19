"""
Test script to plot a waveform using the global reflection model
"""
import matplotlib.pyplot as plt
import numpy as np

from THzProc.THzData import RefData
from base_util import signal_model_functions as sm

filename = 'C:\\Work\\Refs\\ref 21FEB2018\\40ps waveform.txt'

gate = [500, 1250]

n_layers = 1

d = np.array([0.127])

ref = RefData(filename, gate=gate)

e0 = -ref.freq_waveform

theta0 = 17.5 * np.pi / 180

n_ebc = complex(6.7, -0.04)
n_array = np.array([1, n_ebc, np.inf])

theta = np.zeros(n_layers + 2, dtype=complex)
theta[0] = theta0

for i in range(1, n_layers+2):
    theta[i] = sm.get_theta_out(n_array[i-1], n_array[i], theta[i-1])

gamma = sm.global_reflection_model(n_array, theta, ref.freq, d, n_layers)

E2 = gamma[0] * e0

e2 = np.fft.irfft(E2) / ref.dt

plt.figure('Model of EBC on Si')
plt.plot(ref.time, e2, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Ref Test')
plt.subplot(211)
plt.plot(ref.time, ref.waveform, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(212)
plt.plot(ref.freq, np.abs(ref.freq_waveform), 'r')
plt.plot(ref.freq, np.abs(e0), 'b')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, 4)

plt.show()
