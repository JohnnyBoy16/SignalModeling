"""
Test script to check index of refraction solution be using the forward model
"""
import copy

import numpy as np
import matplotlib.pyplot as plt
import wx

from THzProc.THzData import THzData
from THzProc.FrameHolder import FrameHolder
from THzProc.THzData import RefData
import sm_functions as sm
import half_space as hs

# basedir and filename for the reference data
filename = 'C:\\Work\\Refs\\ref 12FEB2018\\100ps waveform.txt'
ref = RefData(filename, zero=True, fix_time=True, gate=[235, 820])

# basedir and filename for the tvl file
basedir = 'F:\\RR 2016\\THz Data\\3-33'
filename = '3-33 Scan 2-12-2018 (res=0.25mm 100ps).tvl'
data = THzData(filename, basedir, follow_gate_on=True, signal_type=1)

# index of refraction value that we want to check
n0 = complex(3.23, -0.1)

# thickness of the sample
d = np.array([1.397])

e0 = -ref.freq_waveform  # initial E0 is the opposite of reference

plt.figure('Reference Data')
plt.subplot(211)
plt.plot(ref.time, ref.waveform, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(ref.freq, np.abs(ref.freq_waveform), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, 3.5)

# build the half space model and add the
theta1 = sm.get_theta_out(1.0, n0, data.theta0)

FSE = hs.half_space_model(e0, data.freq, n0, 0, data.theta0, theta1)
BSE = hs.half_space_model(e0, data.freq, n0, d, data.theta0, theta1)

signal = FSE + BSE

time_signal = np.fft.irfft(signal) / data.dt

plt.figure('Return Signal')
plt.plot(data.time, time_signal, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid(True)

app = wx.App(False)

holder = FrameHolder(data)

plt.show()

app.MainLoop()
