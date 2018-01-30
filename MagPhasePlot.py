import matplotlib.pyplot as plt
import numpy as np

import half_space as hs
import sm_functions as sm


class MagPhasePlot:
    """
    Plots the Magnitude and Phase of the model from a given (nr, ni) location
    and compares that to the magnitude and phase of the actual data waveform
    from the point that was selected on the Index of refraction C-Scan
    """

    def __init__(self, holder, data, e0):

        # The figure and axes on the plot
        self.figure = None
        self.phase_axis = None
        self.mag_axis = None

        # The THzProcClass
        self.data = data

        # the reference signal, already multiplied by -1
        self.e0 = e0

        self.n = complex(1.50, -0.02)

        # the class that holds everything together
        self.holder = holder

        self.is_initialized = False

        self._initialize_figure()

    def _initialize_figure(self):
        """
        Initializes the figure
        """

        self.figure = plt.figure('Magnitude & Phase')
        self.phase_axis = self.figure.add_subplot(211)
        self.mag_axis = self.figure.add_subplot(212)

        self.mag_axis.set_xlabel('Frequency (THz)')

        self.phase_axis.set_ylabel('Phase (rad)')
        self.mag_axis.set_ylabel('Log Magnitude')

        self.phase_axis.grid(True)
        self.mag_axis.grid(True)

    def plot(self, i, j, n=None):
        """
        Plots the magnitude and phase of the transfer functions
        """
        if self.is_initialized:
            for num in range(2):
                self.phase_axis.lines[0].remove()
                self.mag_axis.lines[0].remove()
        else:
            self.is_initialized = True

        if n is None:
            n = self.n

        theta0 = self.data.theta0

        theta1 = sm.get_theta_out(1.0, n, theta0)

        model = hs.half_space_model(self.e0, self.data.freq, n, self.holder.d, theta0, theta1)

        # create the transfer functions that are used to solve the data
        T_model = model / self.e0
        T_data = self.data.freq_waveform[i, j, :] / self.e0

        model_phase = np.unwrap(np.angle(T_model))
        data_phase = np.unwrap(np.angle(T_data))

        model_phase -= model_phase[0]
        data_phase -= data_phase[0]

        model_mag = np.log(np.abs(T_model))
        data_mag = np.log(np.abs(T_data))

        start = self.holder.start_index

        self.phase_axis.plot(self.data.freq[start:], model_phase[start:], 'b', label='Model')
        self.phase_axis.plot(self.data.freq[start:], data_phase[start:], 'r', label='Data')

        self.mag_axis.plot(self.data.freq[start:], model_mag[start:], 'b', label='Model')
        self.mag_axis.plot(self.data.freq[start:], data_mag[start:], 'r', label='Data')

        self.phase_axis.legend()
        self.mag_axis.legend()

        self.figure.suptitle('(%d, %d)' % (i, j))

        # calculate the new limits from the data on the plot
        self.phase_axis.relim()
        self.mag_axis.relim()

        self.phase_axis.autoscale_view()
        self.mag_axis.autoscale_view()

        # draw new lines
        self.figure.canvas.draw()
