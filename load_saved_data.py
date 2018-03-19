"""
Test code index of refraction interactive plot stuff
"""

import pdb
import pickle
import copy

import wx
import matplotlib.pyplot as plt
import numpy as np

from FrameHolder import FrameHolder
from sm_functions import read_reference_data

ref_file = 'C:\\Work\\Refs\\ref 18OCT2017\\30ps waveform.txt'

basedir = 'C:\\Work\\Shim Stock\\New Scans'
tvl_file = 'Yellow Shim Stock.tvl'

ref_time, ref_amp = read_reference_data(ref_file)

d = np.array([0.508])

# index to remove front "blip"
gate0 = 450
# 2nd gate for reference signal, this cuts out on water vapor lines on the
# backside of the reference
gate1 = 2050

ref_freq_amp = copy.deepcopy(ref_amp)
ref_freq_amp[:gate0] = 0
ref_freq_amp[gate1:] = 0
ref_freq_amp[gate0] = ref_freq_amp[gate0+1] / 2
ref_freq_amp[gate1] = ref_freq_amp[gate1-1] / 2

# determine the wave length that was gathered
# calculate dt and df
wave_length = len(ref_time)
dt = ref_time[1]
df = 1 / (wave_length * dt)

ref_freq = np.linspace(0., (wave_length/2) * df, wave_length//2+1)

ref_freq_amp = np.fft.rfft(ref_freq_amp) * dt
e0 = -ref_freq_amp

# load n_array that was solved previously
with open('n_array.pickle', 'rb') as f:
    n_array = pickle.load(f)
f.close()

# load in the results from scipy optimize fmin function
with open('n_array_fmin.pickle', 'rb') as f:
    n_array_fmin = pickle.load(f)
f.close()

# load THzData class
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
f.close()

# load BIG cost array
cost = np.load('cost_array.npy')

app = wx.App(False)

# add theta0 as attribute to data
data.theta0 = 17.5 * np.pi / 180

holder = FrameHolder(data, n_array, n_array_fmin, cost, e0, d)

# TODO figure out a way around this
# may not actually need wxPython for GUI stuff
# close the matplotlib figures that are generated by wxPython

plt.figure('Reference Waveform')
plt.plot(ref_time, ref_amp, 'r')
plt.axvline(ref_time[gate0], linestyle='--', color='k')
plt.axvline(ref_time[gate1], linestyle='--', color='k')
plt.title('Reference Waveform')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Reference Spectrum')
plt.plot(ref_freq, np.abs(ref_freq_amp), 'r')
plt.title('Reference Spectrum')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

# assume that time of flight has been calculated and is in data
plt.figure('Time of Flight')
plt.imshow(data.tof_c_scan, interpolation='none', cmap='gray')
plt.title('Time of Flight')
plt.grid()
plt.colorbar()

plt.show()

app.MainLoop()

