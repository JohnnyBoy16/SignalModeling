import pdb

import numpy as np
from scipy import optimize

import sm_functions as sm
import half_space as hs


def parabolic_equation(data, a, b, c, d, e):
    """
    Equation for a paraboloid to fit to the cost function for determining the
    index of refraction of a material
    :param data:
    :param a:
    :param b:
    :param c:
    :param d:
    :param e:
    :return:
    """
    y, x = data

    paraboloid = ((x - a) / b) ** 2 + ((y - d) / e) ** 2 + c

    return paraboloid


def parameter_gradient_descent(n0, e0, e2, theta0, d, freq, start=0, stop=None,
                               precision=1e-6, max_iter=1e4, gamma=0.01):
    """
    Function to perform a gradient descent search on the cost function for
    material parameter estimation.
    :param n0: The initial guess for the complex index of refraction. The
        imaginary part must be negative to cause extinction
    :param e0: The reference waveform in the frequency domain
    :param e2: The sample waveform in the frequency domain
    :param theta0: The initial angle of the THz beam in radians
    :param d: The thickness of the sample in mm
    :param freq: An array of frequencies over which to calculate the complex
        index of refraction. The array that is returned is the same length as
        freq.
    :param start: The index of the frequency in the frequency array to start
        calculations at. If start is not zero, the points in the return array
        before this index will be (0 -j0)
    :param stop: The index of the frequency in the frequency array to end the
        calculations at.
    :param precision: The change is step size that will terminate the gradient
        descent. Default: 1e-6.
    :param max_iter: The maximum number of iterations at each frequency before
        the gradient descent is terminated. Default: 1e4
    :param gamma: What the error is multiplied by before it is added to the
        current best guess of the solution in the descent. Default: 0.01; This
        is the value that is suggested in [1].
    :return: n_array: The solution for the index of refraction at each
        frequency. n_array is the same length as freq.
    """

    # References
    # [1] "Material parameter estimation with terahertz time domain
    #     spectroscopy", Dorney et al, 2001.
    # [2] "Method for extraction of parameters in THz Time-Domain spectroscopy"
    #     Duvillaret et al, 1996

    if stop is None:
        stop = len(freq)

    n_array = np.zeros(stop, dtype=complex)

    for i in range(start, stop):
        n_sol = n0  # initial guess
        n_iter = 0
        n_step = 100  # reset steps to a large value so it won't stop right away
        k_step = 100
        while (n_step > precision or k_step > precision) and n_iter < max_iter:
            prev_n = n_sol.real
            prev_k = n_sol.imag

            theta1 = sm.get_theta_out(1.0, n_sol, theta0)
            model = hs.half_space_model(e0[:stop], freq[:stop], n_sol, d,
                                        theta0, theta1)

            # transfer functions for the model and data
            T_model = model / e0[:stop]
            T_data = e2[:stop] / e0[:stop]

            # use the unwrapped phase so there are no discontinuities
            data_phase = np.unwrap(np.angle(T_data))
            model_phase = np.unwrap(np.angle(T_model))

            # start the DC phase at zero this is discussed in [1] & [2]
            data_phase -= data_phase[0]
            model_phase -= model_phase[0]

            # use absolute value because phase can be negative
            # this is the error function for phase and magnitude
            rho = (np.abs(data_phase[i]) - np.abs(model_phase)[i])
            phi = np.log(np.abs(T_data))[i] - np.log(np.abs(T_model))[i]

            # adjust the guess
            new_n = prev_n + gamma * rho
            new_k = prev_k + gamma * phi

            # determine how much it changes; when this value is less than
            # precision, loop will end
            n_step = np.abs(new_n - prev_n)
            k_step = np.abs(new_k - prev_k)

            n_sol = complex(new_n, new_k)  # update n_sol

            n_iter += 1

        if n_iter == max_iter:
            print('Max iterations reached at frequency %0.3f!' % freq[i])

        n_array[i] = n_sol  # store solution at that frequency

    return n_array


def scipy_optimize_parameters(data, n0, e0, d, stop_index):

    theta0 = data.theta0

    n_array = np.zeros((data.y_step, data.x_step, stop_index), dtype=complex)

    for i in range(data.y_step):
        print('Row %d of %d' % (i+1, data.y_step))
        for j in range(data.x_step):
            e2 = data.freq_waveform[i, j, :]
            for k in range(stop_index):
                solution = \
                    optimize.fmin(hs.half_space_mag_phase_equation, n0,
                                  args=(e0[:stop_index], e2[:stop_index],
                                        data.freq[:stop_index], d, theta0, k),
                                  disp=False)

                n_array[i, j, k] = complex(solution[0], solution[1])

    return n_array
