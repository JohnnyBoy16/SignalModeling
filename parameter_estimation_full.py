import pdb
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import wx
import skimage.filters

from THzProc.THzData import THzData, RefData
import base_util.signal_model_functions as sm

from FrameHolder import FrameHolder

# the reference file that is to be used in the calculation, must be of the same
# time length and have same wavelength as the tvl data
ref_file = 'C:\\Work\\Refs\\ref 18OCT2017\\30ps waveform.txt'

basedir = 'C:\\Work\\Parameter Estimation\\Shim Stock TVL Files\\New Scans'
tvl_file = 'Yellow Shim Stock.tvl'

# range of real and imaginary values to build the cost function over
nr_bounds = np.linspace(2.5, 1, 50)
ni_bounds = np.linspace(-0.001, -1.0, 50)

# index from which to extract values from tvl file
location = None
# location = np.array([[38, 98],
#                      [90, 32],
#                      [92, 73],
#                      [29, 29],
#                      [61, 65]])

# thickness estimate of the yellow shim stock is 0.508 mm
d = np.array([0.508])  # thickness of the sample in mm, use 0 to look at FSE

c = 0.2998  # speed of light in mm / ps

# gate for Yellow Shim Stock is below
# gate0 = 450
# gate1 = 2050
gate0 = 450  # index to remove front "blip"
gate1 = 2050  # 2nd gate for reference signal, this cuts out on water vapor lines

# gate to initialize the THzData class, want the echo of interest to be in the
# follow gate. This allows us to use peak_bin to gate the area of interest in
# the waveform, a gate of [[100, 1500], [370, 1170]] captures the front surface
# echo in the lead gate and follow gate gets the back surface for the yellow
# shim stock
thz_gate = [[100, 1500], [370, 1170]]

# maximum frequency we want to use in solution
# above this signal to noise level gets too low
max_f = 2.0
min_f = 0.25

# incoming angle of the THz system
theta0 = 17.5 * np.pi / 180

# initial guess for the complex index of refraction of the material
n0 = complex(1.5, -0.2)

# the indices of refraction of the media on each side of the sample under 
# question
n_media = np.array([1.0, 1.0], dtype=complex)

# list of colors to use for plotting
colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']

# ==============================================================================
# END SETTINGS
# START PREPROCESSING

# load the reference signal and tvl scan data
ref = RefData(ref_file, gate=[gate0, gate1])
data = THzData(tvl_file, basedir, gate=thz_gate, follow_gate_on=True)

# determine where the middle of the sample is and adjust the coordinates such
# that (0, 0) is the center of the sample in the C-Scan
# thresh = skimage.filters.threshold_otsu(data.c_scan)
# binary_c_scan = np.zeros(data.c_scan.shape)
# binary_c_scan[np.where(data.c_scan > thresh)] = 1

# sample = np.where(binary_c_scan == 1)
# first_i = sample[0].min()
# last_i = sample[0].max()
# first_j = sample[1].min()
# last_j = sample[1].max()
# center_i = int((last_i + first_i) / 2)
# center_j = int((last_j + first_j) / 2)

# data.adjust_coordinates(center_i, center_j)

# if the sample is NOT overscanned need to comment this out
# index_tuple = data.resize(-12, 12, -12, 12, return_indices=True)

# plot reference waveform and gate0 before modification so we can see what
# it looks like
plt.figure('Reference Waveform')
plt.plot(ref.time, ref.waveform, 'r')
plt.axvline(ref.time[gate0], linestyle='--', color='k')
plt.axvline(ref.time[gate1], linestyle='--', color='k')
plt.title('Reference Waveform')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

# Initial E0 is the negative of the reference frequency due to reflection from
# aluminum plate
e0 = -ref.freq_waveform  # initial E0 is the opposite of reference

plt.figure('Reference Spectrum')
plt.plot(ref.freq, np.abs(ref.freq_waveform), 'r')
plt.title('Reference Spectrum')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

# calculate the frequency index that is closest to max_f
stop_index = np.argmin(np.abs(ref.freq - max_f))
start_index = np.argmin(np.abs(ref.freq - min_f))

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
focus_time = ref.time[ref.waveform.argmax()]

# use the time-shifting property of FFT to align each signal with ref signal
for i in range(data.y_step):
    for j in range(data.x_step):
        hit_time = data.time[data.waveform[i, j, :].argmax()]
        t_diff = focus_time - hit_time
        data.freq_waveform[i, j, :] * np.exp(-2j * np.pi * data.freq * t_diff)

# this needs to be here because there is not a peak_bin_small attribute
if data.has_been_resized:
    i0 = index_tuple[0]
    i1 = index_tuple[1]
    j0 = index_tuple[2]
    j1 = index_tuple[3]
    data.freq_waveform_small = data.freq_waveform[i0:i1, j0:j1, :]

if location is None:
    print('Start Brute Force Search to make Cost Model...')
    t0 = time.time()
    cost = \
        sm.brute_force_search(data.freq_waveform[:, :, :stop_index], 
                              e0[:stop_index], data.freq[:stop_index], 
                              nr_bounds, ni_bounds, n_media, d, theta0, 
                              return_sum=False)

    t1 = time.time()
    print('Brute Force Search Time = %0.4f seconds' % (t1-t0))

t0 = time.time()

if location is None:
    shape = (data.freq_waveform.shape[0], data.freq_waveform.shape[1], stop_index)
    n_array = np.zeros(shape, dtype=complex)
else:
    n_array = np.zeros((len(location), stop_index), dtype=complex)

if location is None:  # solve for every (i, j)
    t0 = time.time()

    print("Starting scipy's optimize")
    n_array_fmin = sm.scipy_optimize_parameters(data, n0, n_media, e0, d, 
                                                stop_index)

    t1 = time.time()
    print('Time for scipy optimize', t1-t0)
    t0 = time.time()

    nrows = data.freq_waveform.shape[0]
    ncols = data.freq_waveform.shape[1]

    print('Starting my gradient descent')
    for i in range(nrows):
        print('Row %d of %d' % (i+1, nrows))
        for j in range(ncols):
            e2 = data.freq_waveform[i, j, :]
            n_array[i, j, :] = \
                sm.parameter_gradient_descent(n0, n_media, e0, e2, theta0, d, 
                                              data.freq, start=start_index, 
                                              stop=stop_index)

else:  # solve only for specified locations
    for loc_num, loc in enumerate(location):
        e2 = data.freq_waveform[loc[0], loc[1], :]
        n_array[loc_num, :] = \
            sm.parameter_gradient_descent(n0, e0, e2, theta0, d, data.freq,
                                          start=start_index, stop=stop_index)

t1 = time.time()
print('Time for gradient descent on true function = %0.4f' % (t1-t0))

# use extent to set the values on the axis label for plotting
extent = (ni_bounds[0], ni_bounds[-1], nr_bounds[-1], nr_bounds[0])

data.make_time_of_flight_c_scan()
plt.figure('Time of Flight')
plt.imshow(data.tof_c_scan, interpolation='none', cmap='gray', 
           extent=data.c_scan_extent)
plt.title('Time of Flight')
plt.grid()
plt.colorbar()

pdb.set_trace()

if location is None:

    app = wx.App(False)

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
