import pdb
import sys
import socket

import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

# handles laptop vs desktop drive issue
if socket.gethostname() == 'Laptop':
    drive = 'C'
else:
    drive = 'D'

sys.path.insert(0, drive + ':\\PycharmProjects\\THzProcClass')

from ParentFrame import ParentFrame


class IndexFrame(ParentFrame):
    """
    Frame to hold the Index of refraction C-Scans for entire sample solution
    """

    def __init__(self, holder):
        """
        Constructor method
        :param holder: The holder that links the frames together. Holder also
            contains n_data information and the THz Waveforms
        """
        # figure and axes for plots
        self.figure = None
        self.real_axis = None
        self.imag_axis = None
        self.holder = holder

        # figure canvas object for drawing updated plots
        self.figure_canvas = None

        super().__init__('Index of Refraction Frame')

        # remove self.axis attribute since it is not needed
        del self.axis

        self.plot()
        self.connect_events()

    # Override from ParentFrame, recreate figure so there are two axes
    def initialize_figure(self):
        """
        Initialize a figure with two subplots
        """
        self.figure = plt.figure()
        self.real_axis = self.figure.add_subplot(211)
        self.imag_axis = self.figure.add_subplot(212)

        self.figure.suptitle(r'Complex $\tilde{n}$ Solution')

        self.figure_canvas = FigureCanvas(self, -1, self.figure)

        plt.close(self.figure)

    def plot(self):
        """
        Plots the point by point index of refraction solution like a C-Scan.
        Real solution is on top and imaginary solution is on the bottom.
        """

        start = self.holder.start_index

        real_c_scan = self.holder.n[:, :, start:].real.mean(axis=2)
        imag_c_scan = self.holder.n[:, :, start:].imag.mean(axis=2)

        real_im = self.real_axis.imshow(real_c_scan, interpolation='none',
                                        cmap='gray')

        imag_im = self.imag_axis.imshow(imag_c_scan, interpolation='none',
                                        cmap='gray')

        # add the colorbar to appropriate axis
        plt.colorbar(real_im, ax=self.real_axis)
        plt.colorbar(imag_im, ax=self.imag_axis)

        self.real_axis.set_ylabel('Index of Refraction')
        self.imag_axis.set_ylabel(r'$\kappa$')

        self.real_axis.grid()
        self.imag_axis.grid()

        # real_cursor = Cursor(self.real_axis, useblit=True, color='red',
        #                      linewidth=1)
        #
        # imag_cursor = Cursor(self.imag_axis, useblit=True, color='red',
        #                      linewidth=1)

        self.figure_canvas.draw()

    def connect_events(self):
        """
        Binds events to methods
        """
        self.figure_canvas.mpl_connect('button_press_event', self.select_point)
        self.figure_canvas.mpl_connect('motion_notify_event', self.motion_handler)

    def select_point(self, event):
        """
        Allows user to click on a point and see corresponding time and frequency
        domain waveforms from the sample along with real and imaginary index of
        refraction solutions vs. frequency.
        """
        xid = event.xdata
        yid = event.ydata

        if event.inaxes:
            j = int(round(xid, 0))
            i = int(round(yid, 0))

            # plot the waveform from point selected
            self.holder.a_scan_frame.plot(i, j)

            # plot the n and kappa vs frequency values for that location
            self.holder.n_scan_frame.plot(i, j)

            # plot the cost function at previous frequency and point selected
            self.holder.cost_frame.plot(i, j)

            # plot the data at the new location clicked and previous model
            self.holder.mag_phase_frame.plot(i, j)

    def motion_handler(self, event):
        """
        Prints the current mouse location and pixel value to the status bar
        """
        if not event.inaxes:
            return

        xid = event.xdata
        yid = event.ydata

        start = self.holder.start_index
        i = int(round(yid, 0))
        j = int(round(xid, 0))

        if event.inaxes == self.real_axis:
            pixel_value = self.holder.n[i, j, start:].real.mean()
        else:
            pixel_value = self.holder.n[i, j, start:].imag.mean()

        status_string = '(%.4f, %.4f), [%.4f]' % (xid, yid, pixel_value)
        self.status_bar.SetStatusText(status_string)
