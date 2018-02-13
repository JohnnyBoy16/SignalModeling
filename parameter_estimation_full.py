import pdb
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import wx

from THzProc.THzData import THzData
from base_util.signal_model_functions import brute_force_search

import sm_functions as sm
import util

from FrameHolder import FrameHolder

# the reference file that is to be used in the calculation, must be of the same
# time length and have same wavelength as the tvl data
ref_file = 'D:\\Work\\Refs\\ref 11DEC2017\\60ps waveform.txt'

basedir = 'F:\\RR 2016\\THz Data\\Grinding Trial Sample\\1st Grind'
tvl_file = 'Sample 4-1 After 1st Polish res=0.25mm.tvl'

# range of real and imaginary values to build the cost function over
nr_bounds = np.linspace(2.5, 1, 50)
ni_bounds = np.linspace(-0.001, -1.0, 50)

# index from which to extract values from tvl file
location = None
# location = np.array([[160, 156],
#                      [255, 217],
#                      [260, 51],
#                      [145, 48],
#                      [50, 50],
#                      [46, 247],
#                      [136, 267]])

# thickness estimate of the yellow shim stock is 0.508 mm
d = np.array([0])  # thickness of the sample in mm, use 0 to look at FSE

c = 0.2998  # speed of light in mm / ps

# gate for Yellow Shim Stock is below
# gate0 = 450
# gate1 = 2050
gate0 = 325  # index to remove front "blip"
gate1 = 910  # 2nd gate for reference signal, this cuts out on water vapor lines

# gate to initialize the THzData class, want the echo of interest to be in the
# follow gate. This allows us to use peak_bin to gate the area of interest in
# the waveform, a gate of [[100, 1500], [370, 1170]] captures the front surface
# echo in the lead gate and follow gate gets the back surface for the yellow
# shim stock
thz_gate = [[100, 1500], [-360, 275]]

# maximum frequency we want to use in solution
# above this signal to noise level gets too low
max_f = 2.5
min_f = 0.25

theta0 = 17.5 * np.pi / 180

# list of colors to use for plotting
colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']

# ==============================================================================
# END SETTINGS
# START PREPROCESSING

# load the reference signal and tvl scan data
ref_time, ref_amp = sm.read_reference_data(ref_file)
data = THzData(tvl_file, basedir, gate=thz_gate, follow_gate_on=True)

index_tuple = data.resize(-12, 12, -12, 12, return_indices=True)

# plot reference waveform and gate0 before modification so we can see what
# it looks like
plt.figure('Reference Waveform')
plt.plot(ref_time, ref_amp, 'r')
plt.axvline(ref_time[gate0], linestyle='--', color='k')
plt.axvline(ref_time[gate1], linestyle='--', color='k')
plt.title('Reference Waveform')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

# remove front blip and add ramp up factor to help with FFT
# create the frequency values that we will actually convert to frequency domain
# in a little bit
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
e0 = -ref_freq_amp  # initial E0 is the opposite of reference

plt.figure('Reference Spectrum')
plt.plot(ref_freq, np.abs(ref_freq_amp), 'r')
plt.title('Reference Spectrum')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

# calculate the frequency index that is closest to max_f
stop_index = np.argmin(np.abs(ref_freq - max_f))
start_index = np.argmin(np.abs(ref_freq - min_f))

# slice out the area around the back surface echo
# using peak_bin prevents us from using numpy slicing though
data.gated_waveform = np.zeros(data.waveform.shape)
for i in range(data.y_step):
    for j in range(data.x_step):
        start = data.peak_bin[3, 1, i, j]
        end = data.peak_bin[4, 1, i, j]
        data.gated_waveform[i, j, start:end] = data.waveform[i, j, start:end]

        # add ramp up factor to help fft
        data.gated_waveform[i, j, start-1] = data.waveform[i, j, start] / 2
        data.gated_waveform[i, j, end] = data.waveform[i, j, end-1] / 2

# calculate frequency domain representation
data.freq_waveform = np.fft.rfft(data.gated_waveform, axis=2) * data.dt

# the arrival time of the reference and data signal may not be exactly the same
# use the time shifting property of the FFT to account for this

# the time of flight at the focus point on reference
focus_time = ref_time[ref_amp.argmax()]

# use the time-shifting property of FFT to align each signal with ref signal
for i in range(data.y_step):
    for j in range(data.x_step):
        hit_time = data.time[data.waveform[i, j, :].argmax()]
        t_diff = focus_time - hit_time
        data.freq_waveform[i, j, :] * np.exp(-2j * np.pi * data.freq * t_diff)

if data.has_been_resized:
    i0 = index_tuple[0]
    i1 = index_tuple[1]
    j0 = index_tuple[2]
    j1 = index_tuple[3]
    data.freq_waveform = data.freq_waveform[i0:i1, j0:j1, :]

if location is None:
    t0 = time.time()

    cost = brute_force_search(data.freq_waveform[:, :, :stop_index], e0[:stop_index],
                              data.freq[:stop_index], nr_bounds, ni_bounds, d, theta0,
                              return_sum=True)

    t1 = time.time()
    print('Brute Force Search Time = %0.4f seconds' % (t1-t0))

t0 = time.time()

# initial guess for the gradient descent search
n0 = complex(1.2, -0.8)
n0_array = np.array([1.2, -0.8])

if location is None:  # search over entire sample
    shape = (data.waveform.shape[0], data.waveform.shape[1], stop_index)
    n_array = np.zeros(shape, dtype=complex)
else:
    n_array = np.zeros((len(location), stop_index), dtype=complex)

if location is None:  # solve for every (i, j)
    t0 = time.time()

    n_array_fmin = util.scipy_optimize_parameters(data, n0_array, e0, d, stop_index)

    t1 = time.time()
    print('Time for scipy optimize', t1-t0)
    t0 = time.time()

    for i in range(data.y_step):
        print('Row %d of %d' % (i+1, data.y_step))
        for j in range(data.x_step):
            # print('Step %d of %d' % (j+1, data.x_step))
            e2 = data.freq_waveform[i, j, :]
            n_array[i, j, :] = \
                util.parameter_gradient_descent(n0, e0, e2, theta0, d, data.freq,
                                                start=start_index, stop=stop_index)

else:  # solve only for specified locations
    for loc_num, loc in enumerate(location):
        e2 = data.freq_waveform[loc[0], loc[1], :]
        n_array[loc_num, :] = \
            util.parameter_gradient_descent(n0, e0, e2, theta0, d, data.freq,
                                            start=start_index, stop=stop_index)

t1 = time.time()
print('Time for gradient descent on true function = %0.4f' % (t1-t0))

# use extent to set the values on the axis label for plotting
extent = (ni_bounds[0], ni_bounds[-1], nr_bounds[-1], nr_bounds[0])

# currently axis labels on mlab figure don't seem to be working
# looks to be a bug on their end
# mlab.figure('Cost Function vs n')
# mlab.surf(ni_bounds, nr_bounds, np.rot90(cost, -1), warp_scale='auto')
# mlab.colorbar()

data.make_time_of_flight_c_scan()
plt.figure('Time of Flight')
plt.imshow(data.tof_c_scan, interpolation='none', cmap='gray')
plt.title('Time of Flight')
plt.grid()
plt.colorbar()

pdb.set_trace()

if location is None:

    app = wx.App(False)

    n_array_fmin = n_array
    holder = FrameHolder(data, n_array, n_array_fmin, cost, e0, d)

    plt.show()

    app.MainLoop()

else:
    plt.figure('Index of Refraction')
    for i in range(len(location)):
        line = n_array[i, start_index:stop_index].real
        plt.plot(data.freq[start_index:stop_index], line, colors[i])
    plt.title('Index of Refraction')
    plt.xlabel('Frequency (THz)')
    plt.ylabel(r'Index of Refraction ($n$)')
    plt.grid()

    plt.figure('Imaginary Index')
    for i in range(len(location)):
        line = n_array[i, start_index:stop_index].imag
        plt.plot(data.freq[start_index:stop_index], line, colors[i])
    plt.title(r'$\kappa$')
    plt.xlabel('Frequency (THz)')
    plt.ylabel(r'$\kappa$')
    plt.grid()

    plt.figure('Data Waveforms')
    for i, loc in enumerate(location):
        plt.plot(data.time, data.waveform[loc[0], loc[1], :], colors[i])
    plt.title('Waveforms')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude')
    plt.grid()

    # remake C-Scan with FSE to show on C-Scan plot with waveform locations
    data.make_c_scan(0)

    plt.figure('C-Scan with Locations')
    im = plt.imshow(data.c_scan, interpolation='none', cmap='gray')
    plt.scatter(y=location[:, 0], x=location[:, 1], color=colors)
    plt.title('C-Scan with Waveform Locations')
    plt.colorbar(im)
    plt.grid()

plt.show()

