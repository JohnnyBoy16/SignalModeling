"""
Test script to check index of refraction solution be using the forward model
"""
import copy
import socket
import sys

import numpy as np
import matplotlib.pyplot as plt
import wx

import half_space as hs
import sm_functions as sm

if socket.gethostname() == 'Laptop':
    drive = 'C:'
else:
    drive = 'D:'

sys.path.insert(0, drive + '\\PycharmProjects\\THzProcClass')

from THzData import THzData
from FrameHolder import FrameHolder

ref_file = drive + '\\Work\\Refs\\ref 18OCT2017\\30ps waveform.txt'

basedir = drive + '\\Work\\Shim Stock\\New Scans'
tvl_file = 'Yellow Shim Stock.tvl'

# index of refraction value that we want to check
n0 = complex(1.64, -0.04)

# thickness of the sample
d = np.array([0.508])

gate0 = 450  # index to remove front "blip"
gate1 = 2050  # 2nd gate for reference signal, this cuts out on water vapor lines

theta0 = 17.5 * np.pi / 180

ref_time, ref_amp = sm.read_reference_data(ref_file)

data = THzData(tvl_file, basedir)

ref_freq_amp = copy.deepcopy(ref_amp)
ref_freq_amp[:gate0] = 0
ref_freq_amp[gate1:] = 0

# determine the wave length that was gathered
# calculate dt and df
wave_length = len(ref_time)
dt = ref_time[1]
df = 1 / (wave_length * dt)

ref_freq = np.linspace(0., (wave_length/2) * df, wave_length//2+1)

ref_freq_amp = np.fft.rfft(ref_freq_amp) * dt
e0 = -ref_freq_amp  # initial E0 is the opposite of reference

plt.figure('Reference Data')
plt.subplot(211)
plt.plot(ref_time, ref_amp, 'r')
plt.axvline(ref_time[gate0], color='k', linestyle='--')
plt.axvline(ref_time[gate1], color='k', linestyle='--')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(212)
plt.plot(ref_freq, np.abs(ref_freq_amp), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xlim(0, 3.5)

# build the half space model and add the
theta1 = sm.get_theta_out(1.0, n0, theta0)

FSE = hs.half_space_model(e0, ref_freq, n0, 0, theta0, theta1)
BSE = hs.half_space_model(e0, ref_freq, n0, d, theta0, theta1)

signal = FSE + BSE

time_signal = np.fft.irfft(signal) / dt

plt.figure('Return Signal')
plt.plot(ref_time, time_signal, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid(True)

# build a global reflection model to see if they are the same
theta = np.zeros(3, dtype=complex)
theta[0] = theta0
theta[-1] = theta0
theta[1] = theta1

n = np.ones(3, dtype=complex)
n[1] = n0
gamma = sm.global_reflection_model(n, theta, data.freq, d, n_layers=1)

signal = gamma[0] * e0
time_signal = np.fft.irfft(signal) / dt

plt.figure('Return Signal from Global Model')
plt.plot(ref_time, time_signal, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid(True)

app = wx.App(False)

holder = FrameHolder(data)

plt.show()

app.MainLoop()
