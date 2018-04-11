"""
Script to test deconvolution on a sample
"""
import copy

import matplotlib.pyplot as plt
import numpy as np

# from clean_decon import hogbom

from THzProc.THzData import THzData, RefData
import base_util.signal_model_functions as sm


def hogbom_decon(waveform, reference, n_iter, gain):
    """
    Implementation of Hogbom's clean deconvolution algorithm
    :param waveform: An A-Scan waveform to deconvolve
    :param n_iter: The maximum number of iterations to perform
    :param gain: the multiplier for value subtraction
    """

    res = copy.deepcopy(waveform)
    comps = np.zeros(waveform.shape)

    for i in range(n_iter):
        argmax = np.abs(res).argmax()
        ref_argmax = reference.argmax()
        diff = argmax - ref_argmax

        # shift the reference signal so its max is at the same point as the data
        shifted_db = np.roll(reference, diff)

        mval = res[argmax] * gain
        comps[argmax] += mval
        res -= shifted_db * mval

        if np.abs(res).max() < 4 * np.abs(res).mean():
            break

    print('Number of iterations', i)

    return res, comps   


# basedir and filename for the reference data
filename = 'C:\\Work\\Refs\\ref 12FEB2018\\100ps waveform.txt'
ref = RefData(filename, zero=True, fix_time=True, gate=[235, 820])

# basedir and filename for the tvl file
basedir = 'F:\\RR 2016\\THz Data\\3-33'
filename = '3-33 Scan 2-12-2018 (res=0.25mm 100ps).tvl'
data = THzData(filename, basedir, follow_gate_on=True, signal_type=1)

# index of refraction of the abradable layer
n0 = complex(2.921, -0.1)

# thickness of the Abradable layer before any grinding
d = np.array([1.397])

# speed of light in mm/ps
c = 0.2998

plt.figure('Reference Data')
plt.subplot(211)
plt.plot(ref.time, ref.waveform, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(212)
plt.suptitle('Reference Waveform')
plt.plot(ref.freq, np.abs(ref.freq_waveform), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, 3.5)

plt.figure('Original Waveform')
plt.plot(data.time, data.waveform[60, 60, :], 'r')
plt.title('Original Waveform')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

# test subtraction of reference waveform from actual data at one point to see
# what it looks like
single_wave = data.waveform[60, 60, :]
single_wave[:246] = 0

FSE = np.zeros(single_wave.shape)
FSE[:820] = single_wave[:820]

interface = np.zeros(single_wave.shape)
interface[1230:2048] = single_wave[1230:2048]

FSE_freq = np.fft.rfft(FSE) * data.dt
interface_freq = np.fft.rfft(interface) * data.dt

plt.figure('Front Surface Echo in Time domain')
plt.subplot(211)
plt.plot(data.time, FSE, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(212)
plt.plot(data.freq, np.abs(FSE_freq), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, 3.5)

omega = data.freq * 2 * np.pi
fc = 3.0  # cutoff frequency in THz
t0 = data.time[single_wave.argmax()]

hanning = np.exp(-1j*omega*t0) * np.cos(omega/(4*fc))**2

# use the np.where function to force all frequency components above fc to be 0
hanning = np.where(data.freq <= fc, hanning, 0)

plt.figure('Hanning Window for FSE')
plt.suptitle('Hanning Window for FSE')
plt.subplot(211)
plt.plot(data.freq, np.abs(hanning), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(data.time, np.fft.irfft(hanning) / data.dt, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

deconvolved_freq = FSE_freq / (ref.freq_waveform) * hanning

deconvolved = np.fft.irfft(deconvolved_freq) / data.dt

plt.figure('Deconvolved FSE')
plt.suptitle('Deconvolved Front Surface Echo')
plt.subplot(211)
plt.plot(data.time, deconvolved, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(212)
plt.plot(data.freq, np.abs(deconvolved_freq), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()

# now work on the interface

fc = 0.7

t0 = data.time[interface.argmax()]

hanning = np.exp(-1j*omega*t0) * np.cos(omega/(4*fc))**2
hanning = np.where(data.freq <= fc, hanning, 0)

# t_diff = data.time[interface.argmax()] - data.time[ref.waveform.argmax()]
t_diff = 2*n0*d / (c*np.cos(data.theta0))

shifted_ref = ref.freq_waveform * np.exp(-2j*np.pi*ref.freq*t_diff)

deconvolved_interface_f = interface_freq / shifted_ref * hanning

deconvolved_interface = np.fft.irfft(deconvolved_interface_f) / data.dt

plt.figure('Reference after Pulse Spreading')
plt.subplot(211)
plt.plot(data.time, np.fft.irfft(shifted_ref)/data.dt, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(data.freq, np.abs(shifted_ref), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Interface Waveform')
plt.suptitle('Interface Signal')
plt.subplot(211)
plt.plot(data.time, interface, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(data.freq, np.abs(interface_freq), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Hanning Window for interface')
plt.suptitle('Hanning Window for interface')
plt.subplot(211)
plt.plot(data.freq, np.abs(hanning), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(data.time, np.fft.irfft(hanning) / data.dt, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Deconvolved Interface Signal')
plt.suptitle('Deconvolved Interface Signal')
plt.subplot(211)
plt.plot(data.time, deconvolved_interface, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(data.freq, np.abs(deconvolved_interface_f), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()

# TRY WIENER DECONVOLUTION

S = FSE_freq  # measured signal
H = ref.freq_waveform  # impulse response of the system

# construct the Wiener Deconvolution Filter
G = np.conj(H) / (np.abs(H)**2 + 0.01)

X_hat = FSE_freq * G

x_hat = np.fft.irfft(X_hat) / data.dt

plt.figure('Wiener Deconvolution Filter')
plt.plot(data.freq, np.abs(G), 'r')
plt.title('Wiener Deconvolution Filter')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Front Surface: Wiener Deconvolution')
plt.subplot(211)
plt.plot(data.time, x_hat, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
plt.plot()
plt.subplot(212)
plt.plot(data.freq, np.abs(X_hat), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()

# Try dividing the combined interface signal with our signal models for E2
n_ab = complex(2.921, -0.1)
n_ebc = complex(3.106, -0.1)

theta0 = data.theta0

theta_ab = sm.get_theta_out(1, n_ab, theta0)
theta_ebc = sm.get_theta_out(n_ab, n_ebc, theta_ab)

E0 = -ref.freq_waveform

t01 = sm.transmission_coefficient(1, n_ab, theta0, theta_ab)
r12 = sm.reflection_coefficient(n_ab, n_ebc, theta_ab, theta_ebc)
t10 = sm.transmission_coefficient(n_ab, 1, theta_ab, theta0)
t_delay = 2 * n_ab*1.397 / (c*np.cos(theta_ab)) 
shift = np.exp(-2j * np.pi * data.freq * t_delay)

# build e2 model
E2 = E0 * t01 * r12 * t10 * shift

e2 = np.fft.irfft(E2) / data.dt

# build E3 model

# get the other coefficients for E3
t12 = sm.transmission_coefficient(n_ab, n_ebc, theta_ab, theta_ebc)
t21 = sm.transmission_coefficient(n_ebc, n_ab, theta_ebc, theta_ab)
r23 = sm.reflection_coefficient(n_ebc, np.inf)

plt.figure('Model of Abradable/EBC Interface')
plt.subplot(211)
plt.plot(data.time, e2, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(data.freq, np.abs(E2), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

t_delay = 2 * n_ebc * 0.127 / (c*np.cos(theta_ebc))
shift = np.exp(-2j * np.pi * data.freq * t_delay)

deconvolved_interface = interface_freq / E2 * hanning

plt.figure('Signal after dividing by E2')
plt.plot(data.time, np.fft.irfft(deconvolved_interface) / data.dt, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

# comps, res = hogbom(single_wave, ref.waveform, True, 0.2, 0.3, 10000)

# gain = 0.25

# ref_normalized = ref.waveform / ref.waveform.max()
# ref_normalized[:246] = 0

# max_iter = 100

# res, comps = hogbom_decon(single_wave, ref_normalized, max_iter, gain)

# plt.figure('Residuals and Comps')
# plt.subplot(211)
# plt.plot(data.time, res, 'r')
# plt.ylabel('Amplitude')
# plt.title('Residuals')
# plt.grid()
# plt.subplot(212)
# plt.plot(data.time, comps, 'r')
# plt.xlabel('Time (ps)')
# plt.ylabel('Amplitude')
# plt.title('Comps')
# plt.grid()

# # take the frequency domain amplitude of the reference signal as the "clean
# # beam". Convolve this with the point source model that we have in comps
# point_map_freq = np.fft.rfft(comps) * data.dt

# clean_map = point_map_freq * ref.freq_waveform

# clean_map = np.fft.irfft(clean_map) / data.dt

# plt.figure('CLEAN: Initial Work')
# plt.plot(data.time, clean_map, 'r')
# plt.title('CLEAN: Initial Work')
# plt.xlabel('Time (ps)')
# plt.ylabel('Amplitude')
# plt.grid()

# # according to the algorithm we are then supposed to add the clean_map that is
# # the point source model convolved with the clean beam model to whatever is 
# # left over in the dirty map
# final_map = clean_map + res

# plt.figure('Combined Map')
# plt.plot(data.time, final_map, 'r')
# plt.title('Final Map')
# plt.xlabel('Time (ps)')
# plt.ylabel('Amplitude')
# plt.grid()

plt.show()
