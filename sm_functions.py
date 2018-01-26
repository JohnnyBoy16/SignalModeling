"""
Module that contains functions used in signal modelling codes
"""
import pdb
import os

import numpy as np
import pandas as pd


def refractive_index(echo_delta_t, thickness, c=0.2998):
    """
    Calculate the refractive index of a substance using the time domain echos
    :param echo_delta_t: The difference in time between two echos (ps)
    :param thickness: The difference in space between what causes the echos (mm)
    :param c: The speed of light in mm/ps
    :return: The refractive index of the substance
    """
    return c / (2 * thickness / echo_delta_t)


def reflection_coefficient(n1, n2, theta1=0.0, theta2=0.0):
    """
    Determine the reflection coefficient of a media transition with parallel polarized light
    :param n1: the refractive index of the media in which coming from
    :param n2: the refractive index of the media in which going to
    :param theta1: the angle of the incident ray, in radians
    :param theta2: the angle of the transmitted ray, in radians
    :return: The reflection coefficient
    """
    num = n1*np.cos(theta2) - n2*np.cos(theta1)
    denom = n1*np.cos(theta2) + n2*np.cos(theta1)
    return num / denom


def transmission_coefficient(n1, n2, theta1=0, theta2=0):
    """
    Determine the transmission coefficient of a media transmission, independent of polarization
    :param n1: the refractive index of the media in which coming from
    :param n2: the refractive index of the media in which going to
    :param theta1: the angle of the incident ray, in radians
    :param theta2: the angle of the transmitted ray, in radians
    :return: The transmission coefficient
    """
    return 2*n1*np.cos(theta1) / (n1*np.cos(theta2) + n2*np.cos(theta1))


def phase_screen_r(h, k, theta=0):
    """
    Adds the phase screen term from Jim Rose's paper for the reflection coefficient
    :param h: The standard deviation or RMS height of the surface in m
    :param k: The angular wavenumber of the incoming beam in meters
    :param theta: The angle of the incoming ray in radians
    :return: The exponential term to be multiplied by the reflection coefficient
    """
    term = np.exp(-2 * h**2 * k**2 * np.cos(theta)**2)
    return term


def phase_screen_t(h, k1, k2, theta1=0, theta2=0):
    """
    Adds the phase screen term from Jim Rose's paper for the transmission coefficient
    :param h: The standard deviation or RMS height of the surface in meters
    :param k1: The angular wavenumber of the incoming beam in meters
    :param k2: The angular wavenumber of the outgoing beam in meters
    :param theta1: The angle of the incoming ray in radians
    :param theta2: The angle of the outgoing ray in radians
    :return: The exponential term that is to be multiplied by the transmission coefficient
    """
    term = np.exp(-0.5*h**2 * (k2*np.cos(theta2) - k1*np.cos(theta1)) ** 2)
    return term


def read_reference_data(filename, basedir=None, shift=True):
    """
    Reads in data from the reference txt file using the pandas library
    :param filename: The path to the reference txt file
    :param basedir: The path to the directory of the file
    :param shift: Whether or not to shift reference time values so the array
        starts at zero. Default=True
    :return: optical_delay: An array of optical delay values (time array)
             ref_amp: The amplitude of the reference signal at a given time
    """

    if basedir is not None:
        filename = os.path.join(basedir, filename)

    # Read in the reference waveform and separate out the optical delay (time)
    # and the reference amplitude
    reference_data = pd.read_csv(filename, delimiter='\t')
    optical_delay = reference_data['Optical Delay/ps'].values
    ref_amp = reference_data['Raw_Data/a.u.'].values

    if shift:
        optical_delay -= optical_delay[0]

    return optical_delay, ref_amp


def get_theta_out(n0, n1, theta0):
    """
    Uses Snell's law to calculate the outgoing angle of light
    :param n0: The index of refraction of the incident media
    :param n1: The index of refraction of the outgoing media
    :param theta0: The angle of the incident ray in radians
    :return: theta1: The angle of the outgoing ray in radians
    """
    # make sure that we do float division
    if type(n1) is int:
        n1 = float(n1)

    return np.arcsin(n0/n1 * np.sin(theta0))


def smooth_exponential_transition(e, delta):
    """
    Smooths out a frequency representation of a zero-padded signal by adding and exponential ramp
    to the front end of the array
    :param e: The frequency domain signal
    :param delta: The spacing between points in the frequency domain
    :return:
    """
    i = 0

    # determine where the first non-zero term is
    # first thing to do is check if array is complex, then make sure real and imaginary terms
    # start and end at the same index
    if np.iscomplexobj(e):  # complex numpy array
        while e[i].real == 0 and e[i].imag == 0:
            i += 1
        if e[i].real == 0 or e[i].imag == 0:
            print(i)
            raise Warning('Real and Imaginary terms do not have the same starting index!')
        start = i  # the index where the data starts

        # determine the last index that contains a non zero term,
        # i.e. start of zero padding
        while e[i].real != 0 and e[i].imag != 0:
            i += 1
        # the last frequency term will always have an imaginary value of zero
        if i != len(e)-1 and e[i].real != 0 or e[i].imag != 0:
            raise Warning('Real and Imaginary terms do not have the same ending index!')
        end = i  # the index where the data ends

    else:  # not a complex numpy array, presumably of type int or float
        while e[i] == 0:
            i += 1
        start = i

        while i < len(e) and e[i] != 0:
            i += 1
        end = i

    amplitude = e[start]

    max_x = delta * start  # x value of first data point
    x = np.arange(0, max_x, delta)

    beta = np.log(0.001)
    alpha = np.zeros(len(x), dtype=complex)

    alpha.real = 1/max_x * (np.log(np.abs(amplitude.real)) - beta)
    alpha.imag = 1/max_x * (np.log(np.abs(amplitude.imag)) - beta)

    y = np.zeros(len(x), dtype=complex)
    y.real = np.sign(amplitude.real) * np.exp(alpha.real * x + beta)
    y.imag = np.sign(amplitude.imag) * np.exp(alpha.imag * x + beta)

    # pdb.set_trace()
    e[:start] = y

    # only adjust last non-zero padded value if there are actually any zero padded values
    # on the end
    if end < len(e) - 1:
        e[end+1] = e[end] / 2

    return e


def global_reflection_model(n, theta, freq, d, n_layers, c=0.2998):
    """
    Calculates the global reflection coefficient given in Orfanidis
    "Electromagnetic Waves & Antennas". The global reflection coefficient is used to solve
    multilayer problems.
    :param n: The index of refraction of each of the layers, including the two half space media on
                either side, expected to be len(n_layers + 2)
    :param d: The thickness of each of the slabs, should be of length n_layer
    :param theta: The angle of the beam in each material including the two media on either side,
                length is n_layers+2
    :param freq: An array of frequencies over which to calculate the coefficient
    :param n_layers: The number of layers in the structure
    :param c: The speed of light (default = 0.2998 mm/ps)
    :return: The global reflection coefficient over the supplied frequency range
    """
    try:
        r = np.zeros((n_layers+1, len(freq)), dtype=complex)
        gamma = np.zeros((n_layers+1, len(freq)), dtype=complex)
    except TypeError:
        r = np.zeros(n_layers+1, dtype=complex)
        gamma = np.zeros(n_layers+1, dtype=complex)

    for i in range(n_layers + 1):
        # determine the local reflection
        r[i] = reflection_coefficient(n[i], n[i + 1], theta[i], theta[i + 1])

    # define the last global reflection coefficient as the local reflection coefficient
    gamma[-1, :] = r[-1]

    # calculate global reflection coefficients recursively
    for i in range(n_layers - 1, -1, -1):
        # delta is Orfanidis eq. 8.1.2, with cosine
        delta = 2 * np.pi * freq / c * d[i] * n[i + 1] * np.cos(theta[i + 1])
        z = np.exp(-2j * delta)
        gamma[i, :] = (r[i] + gamma[i + 1, :] * z) / (1 + r[i] * gamma[i + 1, :] * z)

    return gamma


def reflection_equation(n, theta0, freq, d, n_layers, e0, e1, c=0.2998):
    """
    Sets up an equation to back out the index of refraction of multilayer materials. This function
    assumes that a negative imaginary index of refraction term leads to decay. If you pass in freq
    e0, and e1 as an array, the code should attempt to optimize
    :param n: The guess for the index of refraction of each layer, with real terms in the 1st axis
                and imaginary terms in the second
    :param theta0: The angle of the incoming beam in radians
    :param freq: The frequency range that we are interested in solving for
    :param d: The thickness of each layer in mm
    :param n_layers: The number of layers in the structure
    :param e0: The amplitude of the incoming signal over given frequency range
    :param e1: The amplitude of the experimental signal over given frequency range\
    :param c: the speed of light in mm/ps, default: 0.2998 mm/ps
    :return The difference between the actual signal and the predicted signal
    """
    n_real = np.ones(n_layers + 2)
    n_imag = np.zeros(n_layers + 2)
    n_real[1:n_layers + 1] = n[0]
    n_imag[1:n_layers + 1] = n[1]

    n = n_real + 1j * n_imag

    try:  # solving for multiple frequencies
        # angle in each layer
        theta = np.zeros((n_layers+2, len(freq)), dtype=complex)

    except TypeError:  # if freq isn't actually an array (single value)
        theta = np.zeros(n_layers + 2, dtype=complex)

    # the incoming and outgoing angle are given
    theta[0] = theta0
    theta[-1] = theta0

    for i in range(1, n_layers+1):
        theta[i] = get_theta_out(n[i-1], n[i], theta[i-1])

    # get the global reflection coefficient at each layer
    gamma = global_reflection_model(n, theta, freq, d, n_layers, c)

    eq = e1/e0 - gamma[0]

    return eq


def equation_wrapped(n, theta0, freq, d, n_layers, e0, e1, c=0.2998):
    complex_solution = reflection_equation(n, theta0, freq, d, n_layers, e0, e1, c)

    eq = np.zeros(2 * len(complex_solution))

    # alternate the real and imaginary parts in the array
    eq[0:len(eq):2] = complex_solution.real
    eq[1:len(eq):2] = complex_solution.imag

    return eq


def smooth(y, box_size):
    box = np.ones(box_size) / box_size
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
