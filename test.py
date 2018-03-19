import matplotlib.pyplot as plt
import numpy as np

from THzProc.THzData import RefData
from base_util import signal_model_functions as sm

filename = 'C:\\Work\\Refs\\ref 21FEB2018\\40ps waveform.txt'

gate = [500, 1250]

n_layers = 1

d = np.array([0])

ref = RefData(filename, gate=gate)

e0 = -ref.freq_waveform

theta0 = 17.5 * np.pi / 180

n = np.array([1, np.inf])

c = 0.2998

theta = np.zeros(2, dtype=complex)
theta[0] = theta0
theta[1] = 0

r = np.zeros(2, dtype=complex)
r[1] = sm.reflection_coefficient(n[0], n[1], theta[0], theta[1])

delta = 2 * np.pi * ref.freq/c * d[0] * n[0] * np.cos(theta[0])
z = np.exp(-2j * delta)

gamma = (r[1]+r[0] * z) / (1+r[1]*r[0] * z)

E2 = gamma * e0

e2 = np.fft.irfft(E2) / ref.dt

plt.figure('Return Signal')
plt.plot(ref.time, e2, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.show()
