import matplotlib.pyplot as plt
import numpy as np

import half_space as hs
import sm_functions as sm


class ParentPlot:
    """
    Class that is to be the base of any interactive plots. This should be
    inherited by any interactive plot that needs it
    """

    def __init__(self, subplot_grid=(1, 1), title=None):

        nrows = subplot_grid[0]
        ncols = subplot_grid[1]

        self.figure = None
        self.axis = None

        self._initialize_figure(nrows, ncols, title)

    def _initialize_figure(self, nrows, ncols, title):
        """
        Initialized the figure, if
        :param nrows: The number of rows in the subplot
        :param ncols: The number of columns in the subplot
        :param title: The title of the figure
        """

        # if title is None, normal figure numbering will be used
        self.figure = plt.figure(title)

        if nrows == 1 and ncols == 1:
            self.axis = self.figure.add_subplot(111)
        else:
            self.axis = list()
            for i in range(ncols*nrows):
                ax = self.figure.add_subplot(ncols, nrows, i)
                self.axis.append(ax)


class CostPlot:
    """
    Class to hold the plot of the cost for a given frequency at a given (i, j)
    location
    """

    def __init__(self, holder, data, extent=None, title=None):

        if title is None:
            title = 'Cost C-Scan'

        self.figure = None
        self.axis = None

        # the cost image
        self.image = None

        # location of the minimum cost
        self.dot = None

        # the colorbar
        self.colorbar = None

        # the class that holds the frames and allows them to interact
        self.holder = holder

        # An instance of the THzProcClass
        self.data = data

        # the extent values to pass to imshow(), if extent is None, it will
        # just use the indices as if nothing was passed
        self.extent = extent

        # whether or not an image has been created yet
        self.is_initialized = False

        # initialize i and j to values that won't be used
        self.i = int
        self.j = int

        # initialize f to index closes to 1 THz
        self.f_idx = np.abs(self.data.freq - 1).argmin()

        # set up nr and ni bounds
        self.nr_bounds = np.linspace(2.5, 1, 100)
        self.ni_bounds = np.linspace(-0.001, -1.0, 100)

        self._initialize_figure(title)
        self.connect_events()

    def _initialize_figure(self, title):
        """
        Creates the figure
        :param title: The title of the figure
        """

        self.figure = plt.figure(title)
        self.axis = self.figure.add_subplot(111)

        self.axis.set_xlabel(r'$n_{imag}$')
        self.axis.set_ylabel(r'$n_{real}$')
        self.axis.grid(True)

    def plot(self, i=None, j=None, f=None):
        """
        Plots the cost at the given (i, j) location for a specific frequency.
        If i and j are None, it will plot at the same (x, y) location that was
        clicked on previously.
        :param i: The row to plot from
        :param j: The column to plot from
        :param f: The frequency to plot
        """

        if i is not None:
            self.i = i

        if j is not None:
            self.j = j

        if f is not None:
            self.f_idx = f

        # remove the current image
        if self.image is not None:
            self.image.remove()
            self.dot.remove()
        del self.image

        print(self.i, self.j)

        cost_image = self.holder.cost[self.i, self.j, :, :, self.f_idx]
        min_coords = np.unravel_index(cost_image.argmin(), cost_image.shape)

        self.image = self.axis.imshow(cost_image, extent=self.extent, aspect='auto')
        self.dot = self.axis.scatter(x=self.ni_bounds[min_coords[1]],
                                     y=self.nr_bounds[min_coords[0]], color='r')

        frequency = self.data.freq[self.f_idx]
        title = 'Cost at (%d, %d): %0.2f THz' % (self.i, self.j, frequency)
        self.axis.set_title(title)

        self.figure.canvas.draw()

    def connect_events(self):
        """
        Connects the matpolotlib events to their given method handler
        """
        self.figure.canvas.mpl_connect('button_press_event', self.pick_point)

    def pick_point(self, event):
        """
        Shows the magnitude and phase of the model and data at the selected nr,
        ni value
        """
        if not event.inaxes:
            return

        nr = event.ydata
        ni = event.xdata

        n = complex(nr, ni)

        self.holder.mag_phase_frame.plot(self.i, self.j, n)


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
        # if some lines have already been plotted remove them before plotting
        # new ones
        if self.is_initialized:
            for num in range(2):
                self.phase_axis.lines[0].remove()
                self.mag_axis.lines[0].remove()
        else:
            self.is_initialized = True

        if n is None:
            n = self.n
        else:
            self.n = n

        theta0 = self.data.theta0

        theta1 = sm.get_theta_out(1.0, n, theta0)

        model = hs.half_space_model(self.e0*1.1, self.data.freq, n, self.holder.d,
                                    theta0, theta1)

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
        print(self.data.freq[start])

        self.phase_axis.plot(self.data.freq, model_phase, 'b', label='Model')
        self.phase_axis.plot(self.data.freq, data_phase, 'r', label='Data')

        self.mag_axis.plot(self.data.freq[start:], model_mag[start:], 'b', label='Model')
        self.mag_axis.plot(self.data.freq[start:], data_mag[start:], 'r', label='Data')

        self.phase_axis.legend()
        self.mag_axis.legend()

        self.figure.suptitle('(%d, %d)\tn0: (%0.2f %0.2fj)' % (i, j, self.n.real,
                                                               self.n.imag))

        # calculate the new limits from the data on the plot
        self.phase_axis.relim()
        self.mag_axis.relim()

        self.phase_axis.autoscale_view()
        self.mag_axis.autoscale_view()

        # draw new lines
        self.figure.canvas.draw()
