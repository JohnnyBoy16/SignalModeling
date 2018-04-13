"""
Code to drive the parameter estimation routine
"""
import time

def driver(data, ref, n0, n_media, min_f, max_f, d, locations=None, 
           do_brute_force=False, do_scipy=False, do_gradient_descent=False,
           nr_array=None, ni_array=None):
    """
    Driver function to handle calling the intermediary functions for the
    parameter extraction routine.
    :param data:
    :param ref:
    :param n0:
    :param n_media:
    :param min_f:
    :param max_f:
    :param d: The thickness of the material in mm
    :param locations:
    :param do_brute_force:
    :param do_scipy:
    :param do_gradient_descent:
    :param nr_array:
    :param ni_array:
    """

    if do_brute_force:
        pass  # PUT STUFF HERE
    else:
        cost = None

    if do_scipy:
        pass  # PUT STUFF HERE
    else:
        n_scipy = None

    if do_gradient_descent:
        pass  # PUT STUFF HERE
    else:
        n_gradient = None




