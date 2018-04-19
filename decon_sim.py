"""
Script to learn and showcase the Iterative shrinking algorithm for sparse
deconvolution that is presented in Dong's Thesis Section 4.2
See specifically section 4.2.4 on Numerical and Experimental Verification
"""
import pdb

import numpy as np
import matplotlib.pyplot as plt

from THzProc.THzData import RefData


def deconvolution_L1(wave, ref_wave, lam, n_iter):
    """
    L1 sparse deconvolution algorithm
    """

    n_points = len(wave)

    cost = np.zeros(n_iter)

    H = np.zeros((n_points, n_points))

    # before setting up the convolution matrix we need to shift the reference
    # waveform so the argmax is at the zero position
    ref_wave = np.roll(ref_wave, -ref_wave.argmax())

    for i in range(n_points):
        H[:, i] = np.roll(ref_wave, i)

    tau = 2 / np.abs(np.transpose(H) * H)

    f = wave
    for i in range(n_iter):
        s = np.zeros(n_points)
        for k in range(n_points):
            if f[k] <= -lam*tau:
                s[k] = f[k] + lam*tau
            elif np.abs(f[k]) < lam*tau:
                s[k] = 0
            elif f[k] >= lam*tau:
                s[k] = f[k] - lam*tau
            else:
                raise ValueError('None of the conditions met!')

        f = s * (f - tau*np.transpose(H)*(H*f) - wave) 

        cost[i] = 0.5 * np.sum(np.abs(np.matmul(H, f) - wave)**2) + lam * np.sum(np.abs(wave))

    return f, cost


filename = 'C:\\Work\\Refs\\ref 04APR2018\\50ps waveform.txt'

f1 = np.zeros(4096)
f2 = np.zeros(4096)

# signal with impulses farther apart
f1[1460] = 1
f1[1550] = 1

# signal with impulses closer together
f2[1460] = 1
f2[1485] = 1

ref = RefData(filename, gate=[375, 1485])

e0 = -ref.freq_waveform

# frequency array in radius / sec
omega = ref.freq * 2 * np.pi

# cutoff frequency for the hanning window
fc = 4.0

f1_freq = np.fft.rfft(f1) * ref.dt
f2_freq = np.fft.rfft(f2) * ref.dt

# need to shift reference backwards to zero point before convolution, otherwise
# the multiplication of complex values will result in time-shifting
t_diff = -ref.time[ref.waveform.argmax()]

t_shift = np.exp(-2j * np.pi * ref.freq * t_diff)

shifted_ref = e0 * t_shift

plt.figure('Original Reference')
plt.plot(ref.waveform, 'r')
plt.grid()

plt.figure('Shifted Ref')
plt.plot(np.fft.irfft(shifted_ref) / ref.dt, 'r')
plt.grid()

# convolve the spike waveforms with the impulse response
f1_Y = f1_freq * shifted_ref
f2_Y = f2_freq * shifted_ref

f1_y = np.fft.irfft(f1_Y) / ref.dt
f2_y = np.fft.irfft(f2_Y) / ref.dt

t0_f1 = ref.time[f1_y.argmax()]
t0_f2 = ref.time[f2_y.argmax()]

hanning_f1 = np.exp(-1j*omega*t0_f1) * np.cos(omega/(4*fc))**2
hanning_f2 = np.exp(-1j*omega*t0_f2) * np.cos(omega/(4*fc))**2

hanning_f1 = np.where(ref.freq <= fc, hanning_f1, 0)
hanning_f2 = np.where(ref.freq <= fc, hanning_f2, 0)

t_diff_1 = ref.time[f1_y.argmax() - ref.waveform.argmax()]
t_diff_2 = ref.time[f2_y.argmax() - ref.waveform.argmax()]

t_shift_1 = np.exp(-2j * np.pi * ref.freq * t_diff_1)
t_shift_2 = np.exp(-2j * np.pi * ref.freq * t_diff_2)

# when doing deconvolution the reference signal need to be at the same time as
# the measured signal to avoid any time shifting?
t_diff = ref.time[f1_y.argmax()]
t_shift = np.exp(-2j * np.pi * ref.freq * t_diff)
shifted_ref *= t_shift

# deconvolve the waveforms
f1_X = f1_Y / shifted_ref * hanning_f1
f2_X = f2_Y / shifted_ref * hanning_f2

f1_x = np.fft.irfft(f1_X) / ref.dt
f2_x = np.fft.irfft(f2_X) / ref.dt

plt.figure('Hanning Window')
plt.subplot(211)
plt.plot(ref.freq, np.abs(hanning_f1), 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.grid()
plt.subplot(212)
plt.plot(np.fft.irfft(hanning_f1)/ref.dt, 'r')
plt.xlabel('Data Point')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('F1 & F2')
plt.subplot(211)
plt.plot(f1, 'ko-')
plt.title('F1')
plt.grid()
plt.subplot(212)
plt.plot(f2, 'ko-')
plt.title('F2')
plt.grid()

plt.figure('Measured Signals: Frequency Domain')
plt.subplot(211)
plt.title('F1')
plt.plot(ref.freq, np.abs(f1_Y), 'r')
plt.grid()
plt.subplot(212)
plt.title('F2')
plt.plot(ref.freq, np.abs(f2_Y), 'r')
plt.xlabel('Frequency (THz)')
plt.grid()

plt.figure('Measured Signals: Time Domain')
plt.subplot(211)
plt.title('F1')
plt.plot(f1_y, 'r')
plt.grid()
plt.subplot(212)
plt.title('F2')
plt.plot(f2_y, 'r')
plt.grid()

plt.figure('Deconvolved Signals')
plt.subplot(211)
plt.title('F1')
plt.plot(f1_x, 'bo-')
plt.grid()
plt.subplot(212)
plt.plot(f2_x, 'bo-')
plt.title('F2')
plt.grid()

# try sparse deconvolution

lam = 0.01

wave = f1_y
n_iter = 2
ref_wave = np.fft.irfft(ref.freq_waveform) / ref.dt

n_points = len(wave)

cost = np.zeros(n_iter)

H = np.zeros((n_points, n_points))

# before setting up the convolution matrix we need to shift the reference
# waveform so the argmax is at the zero position
ref_wave = np.roll(ref_wave, -ref_wave.argmax())

for i in range(n_points):
    H[:, i] = np.roll(ref_wave, i)

tau = 1 / np.linalg.norm(np.matmul(np.transpose(H), H), 2)

f = wave
for i in range(n_iter):
    s = np.zeros(n_points)
    for k in range(n_points):
        if f[k] <= -lam*tau:
            s[k] = f[k] + lam*tau
        elif np.abs(f[k]) < lam*tau:
            s[k] = 0
        elif f[k] >= lam*tau:
            s[k] = f[k] - lam*tau
        else:
            raise ValueError('None of the conditions met!')

    Hf = np.matmul(H, f)
    Hf_minus_y = Hf - wave
    f = s * (f - tau*np.matmul(np.transpose(H), Hf_minus_y))

    cost[i] = 0.5 * np.sum(np.abs(np.matmul(H, f) - wave)**2) + lam * np.sum(np.abs(wave))

f1_sparse = f

plt.figure('f1 sparse deconvolution')
plt.plot(f1_sparse, 'k-o')
plt.grid()

plt.show()
