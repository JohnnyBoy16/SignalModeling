import matplotlib.pyplot as plt
import numpy as np


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
