import pdb
import sys
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

import sm_functions as sm
import half_space as hs

# load in THzDataClass that I created
sys.path.insert(0, 'C:\\PycharmProjects\\THzProcClass')
from THzData import THzData

# the reference file that is to be used in the calculation, must be of the same
# time length and have same wavelength as the tvl data
ref_file = 'C:\\Work\Refs\\ref 18OCT2017\\30ps waveform.txt'

basedir = 'C:\\Work\\Shim Stock\\New Scans'
tvl_file = 'Yellow Shim Stock.tvl'

# range of real and imaginary values to build the cost function over
nr_bounds = np.linspace(4.25, 1, 250)
ni_bounds = np.linspace(-0.001, -2.0, 250)

location = np.array([16, 5])  # index from which to extract values from tvl file

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

# adjust ref_time so initial value is 0 ps
ref_time -= ref_time[0]

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

# the cost map over given nr & ni list
cost = np.zeros((len(nr_bounds), len(ni_bounds)))
error = np.zeros(cost.shape, dtype=complex)
arg_T = np.zeros((len(nr_bounds), len(ni_bounds), stop_index))
log_abs_T = np.zeros(arg_T.shape)

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

plt.figure('Sample Spectrum')
plt.plot(data.freq, np.abs(data.freq_waveform[location[0], location[1], :]), 'r')
plt.title('Sample Spectrum')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

# experimental data from a point on the scan
e2 = data.freq_waveform[location[0], location[1], :]

t0 = time.time()

# build the cost function over the bounds
for i, nr in enumerate(nr_bounds):
    # print(i)
    for j, ni in enumerate(ni_bounds):
        n = np.array([nr, ni])

        # currently only solving for a single frequency index
        raw_cost = \
            hs.half_space_mag_phase_equation(n, e0[:stop_index], e2[:stop_index],
                                             ref_freq[:stop_index], d, theta0)[30]

        # try to emulate least squares
        cost[i, j] = np.sum(raw_cost)

cost_min_coords = np.argmin(cost)
cost_min_coords = np.unravel_index(cost_min_coords, cost.shape)

# estimate of n from brute force search
n_hat_real = nr_bounds[cost_min_coords[0]]
n_hat_imag = ni_bounds[cost_min_coords[1]]
n_hat = n_hat_real + 1j * n_hat_imag

t1 = time.time()
print('Brute Force Search Time = %0.4f seconds' % (t1-t0))
print(n_hat)

t0 = time.time()

# try optimizing the true function and see how long it takes
# determine the time of flight between the front and back surface echos
FSE = data.time[data.waveform[location[0], location[1], :].argmax()]
start = data.peak_bin[3, 1, location[0], location[1]]  # get left gate index from peak_bin
BSE = data.time[data.waveform[location[0], location[1], start:].argmax() + start]

# since we know the thickness of the sample, we can use that to get a first estimate of the index
# of refraction, initial estimate assumes no angle of incidence
c_sample = 2*d / (BSE-FSE)  # estimate of speed of light in the material
n0 = c / c_sample  # initial guess for n_real
n0 = complex(n0, 0)
print()
print('Initial Guess =', n0)

max_iter = 10000
precision = 1e-6
gamma = 0.01
n_iter = 0

n0 = complex(1.50, 0)

n_array = np.zeros(stop_index, dtype=complex)

for i in range(4, stop_index):
    n0 = complex(1.50, 0)  # initial guess
    n_step = 100
    k_step = 100
    while (n_step > precision or k_step > precision) and n_iter < max_iter:
        prev_n = n0.real
        prev_k = n0.imag

        theta1 = sm.get_theta_out(1.0, n0, theta0)
        model = hs.half_space_model(e0[:stop_index], ref_freq[:stop_index], n0, d, theta0, theta1)

        T_model = model / e0[:stop_index]
        T_data = e2[:stop_index] / e0[:stop_index]

        model_phase = np.unwrap(np.angle(T_model))
        data_phase = np.unwrap(np.angle(T_data))

        rho = np.abs(data_phase[i]) - np.abs(model_phase[i])
        phi = np.log(np.abs(T_data))[i] - np.log(np.abs(T_model))[i]

        new_n = prev_n + gamma * rho
        new_k = prev_k + gamma * phi

        n_step = np.abs(new_n - prev_n)
        k_step = np.abs(new_k - prev_k)

        n0 = complex(new_n, new_k)  # update n0

        n_iter += 1

    if n_iter == max_iter:
        print('Max iterations reached!')
    else:
        print('Number of iterations =', n_iter)

    n_array[i] = n0  # store solution at that frequency
    n_iter = 0  # reset n_iter

t1 = time.time()
print('Time for gradient descent on true function = %0.4f' % (t1-t0))
print(n0)

pdb.set_trace()

# use extent to set the values on the axis label for plotting
extent = (ni_bounds[0], ni_bounds[-1], nr_bounds[-1], nr_bounds[0])

plt.figure('Cost Function vs n')
im = plt.imshow(cost, aspect='auto', interpolation='none', extent=extent)
plt.scatter(x=n_hat_imag, y=n_hat_real, color='r', label='Brute Force')
plt.scatter(x=n0.imag, y=n0.real, color='c', label='Gradient Descent')
plt.title(r'Cost Function vs $\tilde{n}$')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$n$', fontsize=14)
plt.colorbar(im)
plt.legend()
plt.grid()

# currently axis labels on mlab figure don't seem to be working
# looks to be a bug on their end
mlab.figure('Cost Function vs n')
mlab.surf(ni_bounds, nr_bounds, np.rot90(cost, -1), warp_scale='auto')
mlab.colorbar()

plt.figure('Typical Waveform')
plt.plot(data.time, data.waveform[location[0], location[1], :], 'r')
plt.axvline(data.time[data.peak_bin[3, 1, location[0], location[1]]], linestyle='--', color='g')
plt.axvline(data.time[data.peak_bin[4, 1, location[0], location[1]]], linestyle='--', color='b')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Index of Refraction')
plt.plot(data.freq[4:stop_index], n_array[4:stop_index].real)
plt.title('Index of Refraction')
plt.xlabel('Frequency (THz)')
plt.ylabel(r'Index of Refraction ($n$)')
plt.grid()

plt.figure('Imaginary Index')
plt.plot(data.freq[4:stop_index], n_array[4:stop_index].imag)
plt.title(r'$\kappa$')
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\kappa$')
plt.grid()

plt.show()

