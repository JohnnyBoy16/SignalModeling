"""
Script to gather all of the least squares methods I am working on in one place
"""
import copy
import sys
import time
import pdb

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

import sm_functions as sm
import half_space as hs

sys.path.insert(0, 'C:\\Users\\John Nagel\\Box Sync\\PycharmProjects\\THzProcClass3')
from THzData import THzData

__author__ = 'John Nagel'

# ref_file = 'D:\\Work\\Signal Modeling\\References\\ref 12JUN2017\\60 ps waveform.txt'
# tvl_file = 'D:\\Work\\Signal Modeling\\THz Data\\HDPE Lens\\Smooth Side (res=0.5mm, OD=60ps).tvl'
ref_file = 'C:\\Work\\Signal Modeling\\References\\ref 18OCT2017\\30ps waveform.txt'
tvl_file = 'C:\\Work\\Signal Modeling\\THz Data\\Shim Stock\\New Scans\\Yellow Shim Stock.tvl'

# if we want to solve over the entire sample use None
# location_list = None
# location list for PE Window
# location_list = np.array([[37, 34],
#                           [28, 28],
#                           [45, 45],
#                           [40, 40],
#                           [30, 24],
#                           [11, 34],
#                           [58, 28]])

# location list for yellow shim stock
# location_list = np.array([[8, 32],
#                           [16, 5],
#                           [17, 35],
#                           [9, 17],
#                           [5, 8],
#                           [2, 34],
#                           [7, 3]])
location_list = None

# use flag = 1 for half space model to match nr and ni
#          = 2 for half space model to match magnitude and phase
flag = 2

# diameter = 32  # diameter of the PE disk in mm
diameter = 32
radius = diameter / 2

nr_guess = np.linspace(1.00, 4.25, 50)
ni_guess = np.linspace(-0.001, -0.25, 50)
# nr_guess = np.linspace(1.60, 2.0, 200)
# ni_guess = np.linspace(-0.001, -0.25, 200)

# thickness of layer(s) in mm
# thickness of PE disk is 3.15 mm
# thickness of black shim stock is 0.3175 mm
# thickness of the yellow shim stock is 0.508 mm
d = np.array([0.508])

brute_force_on = True
lsq_on = True

# the number of frequency indices to consider having the same index of refraction
# if step is None only solve for 1 value
step = None
# 30
# minimum frequency that we are interested in
min_f = 0.05  # system response starts at 50 GHz

colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']

# maximum frequency that we are interested in
if flag == 1 or flag == 2:
    # max_f = 1.2
    max_f = 2.5
else:
    max_f = 2.5

# incoming angle of the THz beam
theta0 = 17.5 * np.pi / 180.0

# indices to slice the waveform, gate the front and back surface echos
# this removes the initial "blip" and cuts out the water vapor noise in the middle
# produces a much smoother spectrum
cuts = np.loadtxt('shim_stock_gates.txt', dtype=int)

extent = (ni_guess[0], ni_guess[-1], nr_guess[-1], nr_guess[0])

ref_time, ref_amp = sm.read_reference_data(ref_file)
data = THzData(tvl_file)

# adjust time arrays so initial data point is zero
ref_time -= ref_time[0]

# assume both have the same dt
dt = ref_time[-1] / (len(ref_time) - 1)
df = 1 / (len(ref_time) * dt)
freq = np.linspace(0, len(ref_time) / 2 * df, len(ref_time)//2+1)

e0 = copy.deepcopy(ref_amp)  # e0 will be the negative of reference from aluminum plate
e0[:cuts[0]] = 0  # make everything before cut index zero to remove blip
e0[cuts[0]-1] = e0[cuts[0]] / 2  # add ramp up factor to help FFT

e0_gated = np.zeros(e0.shape)
e0_gated[cuts[0]:cuts[1]] = e0[cuts[0]:cuts[1]]

# slice out the area around the back surface echo and add a ramp factor to help fft
data.gated_waveform = np.zeros(data.waveform.shape)
data.gated_waveform[:, :, cuts[2]:cuts[3]] = data.waveform[:, :, cuts[2]:cuts[3]]
data.gated_waveform[:, :, cuts[2]-1] = data.gated_waveform[:, :, cuts[2]-1] / 2
data.gated_waveform[:, :, cuts[3]] = data.gated_waveform[:, :, cuts[3]-1] / 2

# multiply by -1 to recover original signal, reference was obtained from aluminum plate with
# reflection coefficient assumed to be -1
plt.figure('Reference Waveform: Gated')
plt.plot(ref_time, e0_gated, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

e0_gated = -1 * np.fft.rfft(e0_gated) * dt

data.freq_waveform = np.fft.rfft(data.gated_waveform, axis=2) * data.delta_t

# determine the index of freq that is closest to the minimum frequency value that we are
# interested in
start_index = np.argmin(np.abs(freq - min_f))

# find the index closest to max_f and use it as the end point for calculations
stop_index = np.argmin(np.abs(freq - max_f))  # index closest to max_freq

# zero out everything before that starting index and add in a smooth exponential transition
# this prevents the low frequency data from having a larger sample signal than reference signal
e0_gated[:start_index] = 0.0
e0_gated = sm.smooth_exponential_transition(e0_gated, df)

radius = int(round(radius / data.x_res, 0))  # convert radius from mm to indices

data.freq_waveform[:, :, :start_index] = 0.0

if location_list is not None:  # add starting exponential to waveforms in location list only
    for loc in location_list:
        wave = data.freq_waveform[loc[0], loc[1], :]
        data.freq_waveform[loc[0], loc[1], :] = sm.smooth_exponential_transition(wave, data.delta_f)

else:  # add starting exponential to all waveforms
    for i in range(data.y_step):
        for j in range(data.x_step):
            wave = data.freq_waveform[i, j, :]
            data.freq_waveform[i, j, :] = sm.smooth_exponential_transition(wave, data.delta_f)

# we want to make stop_index of multiple of step for easy looping later on
if step is None:
    # if step is None, we want to solve for a single value of n using all frequencies
    step = stop_index
elif (stop_index % step) == 0:  # if step is already a multiple of stop_index don't adjust
    pass
else:
    # make stop_index an integer multiple of step that is larger than the original stop_index
    stop_index = stop_index + (step - stop_index % step)

# lower and upper bound, used to gate the bins in the for loop below
# this way we are passing in a section of the arrays that are step indices wide
# enforce integer division, so lb and ub are both ints
lb = step // 2  # lower bound
ub = step // 2 + 1  # upper bound

t0 = time.clock()

# time shift each waveform appropriately in the frequency domain. This should line up the argmax
# of each waveform with the reference signal. This will hopefully help stabilize the solution.
# For information on time shifting see Brigham's book "The Fast Fourier Transform" section 3-5.
ref_t0 = ref_time[np.argmax(ref_amp)]
if location_list is None:
    i0 = np.argmax(data.waveform, axis=2)
    t0 = data.time[i0]
    t_diff = t0 - ref_t0
    for i in range(data.y_step):
        for j in range(data.x_step):
            data.freq_waveform[i, j, :] *= np.exp(1j*2*np.pi*data.freq*t_diff[i, j])

else:  # location_list is not None; just time shift each point we are looking at
    for i, loc in enumerate(location_list):
        # try time shifting the FSE of each signal to match reference
        t0 = data.time[np.argmax(data.waveform[loc[0], loc[1], :])]
        t_diff = t0 - ref_t0
        print(t_diff)
        data.freq_waveform[loc[0], loc[1], :] *= np.exp(1j*2*np.pi*data.freq*t_diff)

if flag == 1 or flag == 2:
    # flag == 1: use half space model trying to match nr & ni
    # flag == 2: use half space model trying to match magnitude and phase
    t0 = time.time()
    n_solution, lsq_n, cost, lsq_cost = \
        hs.half_space_main(e0_gated, data, location_list, freq, nr_guess, ni_guess, d, theta0, step,
                           stop_index, lb, ub, flag, brute_force_on, lsq_on, radius)

elif flag == 3 or flag == 4:
    # flag == 3: use Orfanidis model trying to match nr & ni
    # flag == 4: use Orfanidis model trying to match magnitude and phase
    pass

t1 = time.time()
print('Total Time: %0.4f' % (t1 - t0))

data.make_time_of_flight_c_scan()

if location_list is not None:
    # we need to create a plotting array for n_solution that is the same size as the frequency array
    # take the n_solution values and extend them over the appropriate frequency range creating lines
    n_plot = np.zeros((len(location_list), stop_index), dtype=complex)
    for i in range(len(location_list)):
        m = 0
        for j in range(step // 2, stop_index, step):
            n_plot[i, j-lb:j+ub] = n_solution[i, m]
            m += 1

    # we want to plot the n values of each location on one plot
    plt.figure('Real Index of Refraction Solution')
    for i in range(len(location_list)):
        plt.plot(freq[:stop_index], n_plot[i, :].real, colors[i])
    plt.title('Real Index of Refraction Solution')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Real Index of Refraction Solution')
    plt.grid()

    # plot the imaginary solution for each location on one plot
    plt.figure('Imaginary Solution')
    for i in range(len(location_list)):
        plt.plot(freq[:stop_index], n_plot[i].imag, colors[i])
    plt.title('Imaginary Index of Refraction Solution')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Imaginary Solution')
    plt.grid()

    # plot C-Scan with waveform locations to see if there may be grouping
    plt.figure('C-Scan with Locations')
    im = plt.imshow(data.c_scan, cmap='gray', interpolation='none', extent=data.c_scan_extent)
    plt.scatter(x=data.x[location_list[:, 1]], y=data.y[location_list[:, 0]], color=colors)
    plt.title('C-Scan with Locations')
    plt.xlabel('X Scan Location (mm)')
    plt.ylabel('Y Scan Location (mm)')
    plt.colorbar(im)
    plt.grid()

    plt.figure('C-Scan with Locations in Indices')
    im = plt.imshow(data.c_scan, cmap='gray', interpolation='none')
    plt.scatter(x=location_list[:, 1], y=location_list[:, 0], color=colors)
    plt.title('C-Scan with Locations')
    plt.xlabel('Column (j)')
    plt.ylabel('Row (i)')
    plt.colorbar(im)
    plt.grid()

    plt.figure('Time of Flight C-Scan')
    im = plt.imshow(data.tof_c_scan, cmap='gray_r', interpolation='none', extent=data.c_scan_extent)
    plt.scatter(x=data.x[location_list[:, 1]], y=data.y[location_list[:, 0]], color=colors)
    plt.title('Time of Flight (ps)')
    plt.xlabel('X Scan Location (mm)')
    plt.ylabel('Y Scan Location (mm)')
    plt.colorbar(im)
    plt.grid()

    # plot the cost function for each location
    for i in range(len(location_list)):
        min_coords = np.argmin(cost[i, :, :, 0])
        min_coords = np.unravel_index(min_coords, cost[i, :, :, 0].shape)

        fig_string = 'Cost Function at [%d, %d]' % (location_list[i][0], location_list[i][1])
        plt.figure(fig_string)
        im = plt.imshow(cost[i, :, :, 0], interpolation='none', extent=extent, aspect='auto')
        if brute_force_on:
            plt.scatter(x=ni_guess[min_coords[1]], y=nr_guess[min_coords[0]], color='r',
                        label='Brute Force')
        if lsq_on:
            plt.scatter(x=lsq_n[i, 1], y=lsq_n[i, 0], color='y', label='LSQ Solution')
        plt.title(fig_string, color=colors[i])
        plt.xlabel(r'$n_{real}$')
        plt.ylabel(r'$n_{imag}$')
        plt.colorbar(im)
        plt.legend()
        plt.grid()

    # plot the waveform from each location
    for i, loc in enumerate(location_list):
        fig_string = 'Waveform at [%d, %d]' % (loc[0], loc[1])
        plt.figure(fig_string)
        plt.plot(data.time, data.waveform[loc[0], loc[1], :], 'r', label='Raw')
        plt.plot(data.time, data.gated_waveform[loc[0], loc[1], :], 'b', label='Gated')
        plt.title(fig_string, color=colors[i])
        plt.xlabel('Time (ps)')
        plt.ylabel('Amplitude')
        plt.grid()

    # plot the reference waveform
    plt.figure('Reference Waveform')
    plt.plot(ref_time, ref_amp, 'r')
    plt.title('Reference Waveform')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude')
    plt.grid()

    mlab.figure('Cost Function Example in 3D')
    mlab.surf(cost[1, :, :, 0], warp_scale='auto')
    mlab.colorbar()

if location_list is None:
    for i in range(stop_index // step):
        # vmax = brute_force_n[:, :, i].real.max()
        # vmin = brute_force_n[brute_force_n[:, :, i].real.nonzero()].min()
        fig_string = 'Real Test %d' % i
        plt.figure(fig_string)
        plt.imshow(n_solution[:, :, i].real, interpolation='none', vmin=1.0,
                   extent=data.c_scan_extent)
        plt.xlabel('X Scan Location')
        plt.ylabel('Y Scan Location')
        plt.grid()
        plt.colorbar()

        fig_string = 'Imaginary Test %d' % i
        plt.figure(fig_string)
        plt.imshow(n_solution[:, :, i].imag, interpolation='none', extent=data.c_scan_extent)
        plt.xlabel('X Scan Location (mm)')
        plt.ylabel('Y Scan Location (mm)')
        plt.grid()
        plt.colorbar()

        plt.figure('Histogram of Real Values')
        plt.hist(n_solution[:, :, i].real.flatten())
        plt.xlabel('n solution')
        plt.ylabel('Number of Values')
        plt.title('Histogram of Real Solution Values')

        plt.figure('Histogram of Imaginary Values')
        plt.hist(n_solution[:, :, i].imag.flatten())
        plt.xlabel(r'$\kappa$ Solution')
        plt.ylabel('Number of Values')
        plt.title('Histogram of Imaginary Solution Values')

plt.show()
