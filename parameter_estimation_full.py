import pdb
import sys
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import wx
from scipy import optimize

import sm_functions as sm
import half_space as hs
import util
from FrameHolder import FrameHolder

# load in THzDataClass that I created
sys.path.insert(0, 'D:\\PycharmProjects\\THzProcClass')
from THzData import THzData

# the reference file that is to be used in the calculation, must be of the same
# time length and have same wavelength as the tvl data
ref_file = 'D:\\Work\Refs\\ref 18OCT2017\\30ps waveform.txt'

basedir = 'D:\\Work\\Shim Stock\\New Scans'
tvl_file = 'Yellow Shim Stock.tvl'

# range of real and imaginary values to build the cost function over
nr_bounds = np.linspace(2.5, 1, 100)
ni_bounds = np.linspace(-0.001, -1.0, 100)

# index from which to extract values from tvl file
location = None
# location = np.array([[16, 5],
#                      [13, 15],
#                      [9, 23],
#                      [2, 1],
#                      [17, 34],
#                      [3, 34],
#                      [6, 7]])

d = np.array([0.508])  # thickness of the yellow shim stock in mm

c = 0.2998  # speed of light in mm / ps

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
max_f = 2.5

start_index = 7

# incoming angle of system is 17.5 degrees
# convert to radians
theta0 = 17.5 * np.pi / 180

# list of colors to use for plotting
colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']

# ==============================================================================
# END SETTINGS
# START PREPROCESSING

# load the reference signal and tvl scan data
ref_time, ref_amp = sm.read_reference_data(ref_file)
data = THzData(tvl_file, basedir, gate=thz_gate, follow_gate_on=True)

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
step = stop_index

lb = step // 2  # lower bound
ub = step // 2 + 1  # upper bound

# slice out the area around the back surface echo
# using peak_bin prevents us from using numpy slicing though
# TODO perhaps not, look into advanced numpy slicing
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

# experimental data from a point on the scan
if location is None:
    e2 = data.freq_waveform[16, 5, :]
else:
    e2 = data.freq_waveform[location[0, 0], location[0, 1], :]

# the cost map is 5D, [i, j, nr, ni, freq]
size = (data.y_step, data.x_step, len(nr_bounds), len(ni_bounds), stop_index)
cost = np.zeros(size)

t0 = time.time()

# calculate the cost function for each (i, j) pair over the nr, ni guess range
for y in range(data.y_step):
    print('y %d of %d' % (y+1, data.y_step))
    for x in range(data.x_step):
        # print('x %d of %d' % (x+1, data.x_step))
        for i, nr in enumerate(nr_bounds):
            for j, ni in enumerate(ni_bounds):
                n = np.array([nr, ni])

                # currently only solving for a single frequency index
                raw_cost = \
                    hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                                     ref_freq[:stop_index], d, theta0)

                # store cost
                cost[y, x, i, j, :] = raw_cost

t1 = time.time()
print('Brute Force Search Time = %0.4f seconds' % (t1-t0))

t0 = time.time()

# initial guess for the gradient descent search
n0 = complex(1.2, -0.8)
n0_array = np.array([1.2, -0.8])

if location is None:  # search over entire sample
    shape = (data.waveform.shape[0], data.waveform.shape[1], stop_index)
    n_array = np.zeros(shape, dtype=complex)
    n_array_fmin = np.zeros(shape, dtype=complex)
else:
    n_array = np.zeros((len(location), stop_index), dtype=complex)

if location is None:  # solve for every (i, j)
    for i in range(data.y_step):
        print('Row %d of %d' % (i+1, data.y_step))
        for j in range(data.x_step):
            e2 = data.freq_waveform[i, j, :]
            for k in range(stop_index):
                solution = \
                    optimize.fmin(hs.half_space_mag_phase_equation, n0_array,
                                  args=(e0[:stop_index], e2[:stop_index],
                                        data.freq[:stop_index], d, theta0, k),
                                  disp=False)

                n_array_fmin[i, j, k] = complex(solution[0], solution[1])
    t1 = time.time()
    print('Time of scipy.optimize = %0.4f' % (t1-t0))
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

if location is None:

    app = wx.App(False)

    holder = FrameHolder(data, n_array, n_array_fmin, cost, e0, d)

    plt.show()

    app.MainLoop()

else:
    plt.figure('Index of Refraction')
    for i in range(len(location)):
        plt.plot(data.freq[start_index:stop_index], n_array[i, 4:stop_index].real, colors[i])
    plt.title('Index of Refraction')
    plt.xlabel('Frequency (THz)')
    plt.ylabel(r'Index of Refraction ($n$)')
    plt.grid()

    plt.figure('Imaginary Index')
    for i in range(len(location)):
        plt.plot(data.freq[start_index:stop_index], n_array[i, 4:stop_index].imag, colors[i])
    plt.title(r'$\kappa$')
    plt.xlabel('Frequency (THz)')
    plt.ylabel(r'$\kappa$')
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
