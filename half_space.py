import pdb
import copy

import numpy as np
from scipy.optimize import leastsq, minimize

import sm_functions as sm
import util


def brute_force_search(e0, e2, freq, nr_list, ni_list, d, theta0, step, stop_index, lb, ub,
                       c=0.2998):
    """
    Manually searches over the given range of real and imaginary index of refraction values to
    build a 2D map of the solution space
    :return: the cost map for the given nr & ni values
    """

    # the cost map over given nr & ni list
    cost = np.zeros((len(nr_list), len(ni_list), stop_index//step))

    for i, nr in enumerate(nr_list):
        for j, ni in enumerate(ni_list):
            m = 0  # use a different counter for cost array
            n = np.array([nr, ni])

            for k in range(step//2, stop_index, step):
                raw_cost = half_space_mag_phase_equation(n, e0[k-lb:k+ub], e2[k-lb:k+ub],
                                                         freq[k-lb:k+ub], d, theta0, c)

                # try to emulate least squares
                cost[i, j, m] = np.sum(raw_cost)
                m += 1

    return cost


def half_space_model(e0, freq, n, d, theta0, theta1, c=0.2998):
    """
    Uses the half space model that does not include the Fabry-Perot effect.
    Creates the model signal by chaining together the Fresnel reflection coefficients, eg.
    E1 = E0*T01*R12*T10 + E0*T01*T12*R23*T21*T10*exp(...). Where exp(...) is the propagation factor.
    """
    t01 = sm.transmission_coefficient(1.0, n, theta0, theta1)
    t10 = sm.transmission_coefficient(n, 1.0, theta1, theta0)

    r01 = sm.reflection_coefficient(1.0, n, theta0, theta1)
    r10 = sm.reflection_coefficient(n, 1.0, theta1, theta0)

    # t_delay also includes imaginary n value, so signal should decrease
    t_delay = 2 * n*d / (c*np.cos(theta1))  # factor of two accounts for back and forth travel

    shift = np.exp(-1j * 2*np.pi * freq * t_delay)

    # is distance is given as 0, we just want to look at FSE reflection
    if d == 0:
        model = e0 * r01
    else:
        model = e0 * t01 * r10 * t10 * shift

    factor = util.gaussian_1d(d/n.real, 1, 0, 1.22)

    model *= factor

    return model


def half_space_mag_phase_equation(n_in, e0, e2, freq, d, theta0, k=None, c=0.2998):
    """
    Function wrapper for the half space model that compares the magnitude and phase of the model
    that is derived from the reference signal (e0) to the experimental data (e2)
    """
    # scipy doesn't like to deal with complex values, so let n_in be a 2 element
    # array and then make it complex
    n_out = n_in[0] + 1j * n_in[1]

    # determine the angle in the material for whatever the index of refraction is
    theta1 = sm.get_theta_out(1.0, n_out, theta0)

    # build the model
    model = half_space_model(e0, freq, n_out, d, theta0, theta1, c)

    T_model = model / e0
    T_data = e2 / e0

    # unwrap the phase so it is a continuous function
    # this makes for easier solving numerically
    model_phase = np.unwrap(np.angle(T_model))
    e2_phase = np.unwrap(np.angle(T_data))

    # set the DC phase to zero
    model_phase -= model_phase[0]
    e2_phase -= e2_phase[0]

    # add in the error function that is found in Duvillaret's 1996 paper
    rho = np.log(np.abs(T_data)) - np.log(np.abs(T_model))
    phi = e2_phase - model_phase

    delta = rho**2 + phi**2

    if k is None:
        return delta
    else:
        return delta[k]


def least_sq_wrapper(n_in, e0, e2, freq, d, theta0, c=0.2998):
    """
    Equation that is to be used with scipy's least_sq function to estimate a
    sample's material parameters numerically
    :param n_in: An array with two values, first value is the real part of the
        complex n, the second value is the imaginary part. Let the imaginary
        part be negative for extinction.
    :param e0: The reference signal in the frequency domain
    :param e2: The sample signal in the frequency domain
    :param freq: The frequency array over which to be solved
    :param d: The thickness of the sample in mm
    :param theta0: The initial angle of the THz beam in radians
    :param c: The speed of light in mm/ps (default: 0.2998)
    :return:
    """

    n = complex(n_in[0], n_in[1])

    # determine the ray angle of the THz beam in the material
    theta1 = sm.get_theta_out(1.0, n, theta0)

    # build the model
    model = half_space_model(e0, freq, n, d, theta0, theta1, c)

    # create transfer function to try and remove system artifacts
    T_model = model / e0
    T_data = e2 / e0

    # unwrap the phase so it is a continuous function
    # this makes for easier solving numerically
    model_unwrapped_phase = np.unwrap(np.angle(T_model))
    e2_unwrapped_phase = np.unwrap(np.angle(T_data))

    model_mag = np.abs(T_model)
    e2_mag = np.abs(T_data)

    mag_array = np.log(e2_mag) - np.log(model_mag)
    phase_array = e2_unwrapped_phase - model_unwrapped_phase

    return_array = np.r_[mag_array, phase_array]

    return return_array
