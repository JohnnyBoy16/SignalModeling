import pdb
import copy

import numpy as np
from scipy.optimize import leastsq, minimize

import sm_functions as sm


def half_space_main(e0, data, location_list, freq, nr_list, ni_list, d, theta0, step,
                    stop_index, lb, ub, brute_force_on, lsq_on, c=0.2998):

    if location_list is None:
        lsq_n = np.zeros((data.y_step, data.x_step, stop_index//step), dtype=complex)
        lsq_cost = np.zeros((data.y_step, data.x_step))
        brute_force_n = np.zeros((data.y_step_small, data.x_step_small, stop_index//step),
                                 dtype=complex)
    else:  # location list is None, search every location
        lsq_n = np.zeros((len(location_list), stop_index//step), dtype=complex)
        lsq_cost = np.zeros(len(location_list))
        brute_force_n = np.zeros((len(location_list), stop_index//step), dtype=complex)

    # find optimal complex index of refraction only at selected points in location list
    if lsq_on and location_list is not None:
        for i, loc in enumerate(location_list):
            n0 = np.array([2.0, 0.2])
            e2 = copy.deepcopy(data.freq_waveform[loc[0], loc[1], :])

            sol = leastsq(half_space_mag_phase_equation, n0, args=(e0, e2, freq, d, theta0))

            lsq_n[i] = sol[0][0] + 1j*sol[0][1]
            # lsq_cost[i] = sol.cost

    # perform a point-by-point least squares minimization to find the optimal real and
    # imaginary parts of the index of refraction
    elif lsq_on and location_list is None:
        for i in range(data.y_step):
            for j in range(data.x_step):
                n0 = np.array([2.0, -0.2])
                e2 = copy.deepcopy(data.freq_waveform[i, j, :])
                sol = leastsq(half_space_mag_phase_equation, n0,
                              args=(e0, e2, freq, d, theta0))

                # according to scipy docs, sol[1] is an integer flag and if it is larger than
                #  4, a solution was not found
                if sol[1] > 4:
                    raise ValueError('Solution Not Found')

                lsq_n[i, j] = sol[0][0] + 1j*sol[0][1]

    if brute_force_on and location_list is not None:
        # if location list is not None just do a search over the given points of interest
        cost = np.zeros((len(location_list), len(nr_list), len(ni_list), stop_index//step))
        for l, loc in enumerate(location_list):
            print('Location %d of %d' % (l+1, len(location_list)))
            e2 = copy.deepcopy(data.freq_waveform[loc[0], loc[1], :])
            cost[l] = brute_force_search(e0, e2, freq, nr_list, ni_list, d, theta0, step,
                                         stop_index, lb, ub, c)

        # if location list is None perform a search over all points
    elif brute_force_on and location_list is None:
        cost = np.zeros((data.y_step_small, data.x_step_small, len(nr_list), len(ni_list),
                         stop_index//step))

        for y in range(data.y_step_small):
            print('Row %d of %d' % (y+1, data.y_step_small))
            for x in range(data.x_step_small):
                # print('Point %d of %d' % (x+1, data.x_step_small))

                e2 = copy.deepcopy(data.freq_waveform[y, x, :])
                cost[y, x, :, :, :] = brute_force_search(
                    e0, e2, freq, nr_list, ni_list, d, theta0, step, stop_index, lb, ub, c)

    # if we performed a brute force search, we know have the cost function for all search
    # locations, but we need to know the minimum value location
    if brute_force_on and location_list is not None:
        # determine the minimum cost value in each blanket search
        # the range over which the blanket search occurs
        for l in range(len(location_list)):
            for k in range(stop_index // step):
                min_coords = np.argmin(cost[l, :, :, k])
                # calculate min location and use unravel_index to give correct coordinates
                min_coords = np.unravel_index(min_coords, cost[l, :, :, k].shape)
                brute_force_n[l, k] = complex(nr_list[min_coords[0]], ni_list[min_coords[1]])

    elif brute_force_on and location_list is None:
        # determine the minimum cost of each brute force search
        for y in range(data.y_step_small):
            for x in range(data.x_step_small):
                for k in range(stop_index // step):
                    min_coords = np.argmin(cost[y, x, :, :, k])
                    # calculate min location and use unravel_index to give correct coordinates
                    min_coords = np.unravel_index(min_coords, cost[y, x, :, :, k].shape)
                    brute_force_n[y, x, k] = complex(nr_list[min_coords[0]], ni_list[min_coords[1]])

    return brute_force_n, lsq_n, cost, lsq_cost


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

    r10 = sm.reflection_coefficient(n, 1.0, theta1, theta0)
    r10 = -1  # account for perfect reflection off of substrate

    # t_delay also include imaginary n value, so signal should decrease
    t_delay = 2 * n*d / (c*np.cos(theta1))  # factor of two accounts for back and forth travel

    shift = np.exp(-1j * 2*np.pi * freq * t_delay)

    model = e0 * t01 * r10 * t10 * shift

    return model


def half_space_model_equation(n_in, e0, e2, freq, d, theta0, c=0.2998):
    """
    Function wrapper for the half space model that compares the real and imaginary parts of the
    model transmission coefficient that is derived from the comparison of the reference signal
    (e0) to the experimental data (e2)
    """

    n_in[1] *= -1

    delta = half_space_mag_phase_equation(n_in, e0, e2, freq, d, theta0, c)

    return np.sum(delta)


def half_space_mag_phase_equation(n_in, e0, e2, freq, d, theta0, c=0.2998):
    """
    Function wrapper for the half space model that compares the magnitude and phase of the model
    that is derived from the reference signal (e0) to the experimental data (e2)
    """
    # scipy doesn't like to deal with complex values, so let n_in be a 2 element array and then
    # make it complex
    n_out = n_in[0] + 1j * n_in[1]

    # determine the angle in the material for whatever the index of refraction is
    theta1 = sm.get_theta_out(1.0, n_out, theta0)

    # build the model
    model = half_space_model(e0, freq, n_out, d, theta0, theta1, c)

    # model[0] = 0.0  # assume 0 DC frequency, good idea????

    T_model = model / e0
    T_data = e2 / e0

    # unwrap the phase so it is a continuous function
    # this makes for easier solving numerically
    model_unwrapped_phase = np.unwrap(np.angle(T_model))
    e2_unwrapped_phase = np.unwrap(np.angle(T_data))

    # add in the error function that is found in Duvillaret's 1996 paper
    rho = np.log(np.abs(T_data)) - np.log(np.abs(T_model))
    phi = e2_unwrapped_phase - model_unwrapped_phase

    # delta = np.array([rho, phi])
    delta = rho**2 + phi**2

    # return the model unwrapped phase to see if it forms a plane versus (nr, ni)
    return delta  # , model
