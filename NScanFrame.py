import pdb
import sys
import socket

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

# handles laptop vs desktop drive issue
if socket.gethostname() == 'Laptop':
    drive = 'C'
else:
    drive = 'D'

sys.path.insert(0, drive + ':\\PycharmProjects\\THzProcClass')

from ParentFrame import ParentFrame


class NScanFrame(ParentFrame):
    """
    Displays the real and imaginary parts of the index of refraction vs.
    frequency for the point that was clicked on
    """

    def __init__(self, holder, data):
        """
        Constructor method
        :param holder: A linker class that holds the frames together
        :param data: An instance of THzProcClass
        """

        # figure and axes for the plots
        self.figure = None
        self.real_axis = None
        self.imag_axis = None

        # figure canvas that is used for drawing the plots
        self.figure_canvas = None

        super().__init__('N-Scan Frame')

        # a holder class that links the plots
        self.holder = holder

        self.data = data

        # keeps track of whether a point in the index of refraction C-Scan has
        # been clicked on and the index of refraction solution has been plotted
        # for the first time
        self.is_initialized = False

        # store the line that was clicked on
        self.line_held = None

        # keep track of the last waveform index that was plotted
        self.i = int  # know that these will be of type int
        self.j = int

        # initialize f to index closes to 1 THz
        self.f_idx = np.abs(self.data.freq - 1).argmin()

        self.connect_events()

        plt.close()

    # Override from ParentFrame, want to have two subplots instead of one
    def initialize_figure(self):
        """
        Initializes a figure with two subplots
        """
        self.figure = plt.figure()
        self.real_axis = self.figure.add_subplot(211)
        self.imag_axis = self.figure.add_subplot(212)

        self.figure_canvas = FigureCanvas(self, -1, self.figure)

    def plot(self, i=None, j=None, f=None):
        """
        Plots the real and imaginary parts of the index of refraction solution
        """
        if not self.is_initialized:
            self.is_initialized = True

        if i and j is not None:
            self.i = i

        if j is not None:
            self.j = j

        if f is not None:
            self.f_idx = f

        start = self.holder.start_index

        self.real_axis.cla()
        self.imag_axis.cla()

        # plot the real and imaginary solution from my gradient of descent
        # function
        real_line = self.holder.n[self.i, self.j, start:].real
        imag_line = self.holder.n[self.i, self.j, start:].imag
        self.real_axis.plot(self.data.freq[start:], real_line, label='My Solution')
        self.imag_axis.plot(self.data.freq[start:], imag_line, label='My Solution')

        # plot the real and imaginary solution from scipy optimize fmin
        real_line = self.holder.nfmin[self.i, self.j, start:].real
        imag_line = self.holder.nfmin[self.i, self.j, start:].imag
        self.real_axis.plot(self.data.freq[start:], real_line, 'r', label='Fmin Solution')
        self.imag_axis.plot(self.data.freq[start:], imag_line, 'r', label='Fmin Solution')

        self.real_axis.axvline(self.data.freq[self.f_idx], linestyle='--',
                               color='k', picker=2, linewidth=1.0)
        self.imag_axis.axvline(self.data.freq[self.f_idx], linestyle='--',
                               color='k', picker=2, linewidth=1.0)

        self.real_axis.legend()
        self.imag_axis.legend()

        self.real_axis.set_ylabel('Real Solution')
        self.imag_axis.set_ylabel('Imaginary Solution')

        self.imag_axis.set_xlabel('Frequency (THz)')

        self.real_axis.grid()
        self.imag_axis.grid()

        self.figure_canvas.draw()

    def connect_events(self):
        self.figure_canvas.mpl_connect('motion_notify_event', self.motion_handler)
        self.figure_canvas.mpl_connect('pick_event', self.grab_line)
        self.figure_canvas.mpl_connect('motion_notify_event', self.slide_line)
        self.figure_canvas.mpl_connect('button_release_event', self.release_line)

    def motion_handler(self, event):
        """
        Prints the current (x, y) location to the status bar
        """
        if not self.is_initialized:
            return

        if not event.inaxes:
            self.status_bar.SetStatusText('')
            return

        xid = event.xdata
        yid = event.ydata

        status_string = '(%0.4f, %0.4f)' % (xid, yid)
        self.status_bar.SetStatusText(status_string)

    def grab_line(self, event):
        """
        Determines what line was clicked on
        """

        # if no point has been clicked on in the index of refraction C-Scan yet,
        # do nothing and return
        if not self.is_initialized:
            return

        if not isinstance(event.artist, Line2D):
            return

        # store the line that was grabbed
        self.line_held = event.artist

        # make the line bold for easy viewing
        self.line_held.set_linewidth(2.0)

        self.figure_canvas.draw()  # update figure

    def slide_line(self, event):
        """
        Slides the line along with the mouse after it has been grabbed
        """
        if not self.is_initialized:
            return

        if self.line_held is None:
            return

        if not event.inaxes:
            return

        self.line_held.set_xdata([event.xdata, event.xdata])
        self.figure_canvas.draw()

    def release_line(self, event):
        """
        Determines what frequency will be displayed in the cost plot based on
        where the line was released
        """
        # if no point on the index of refraction C-Scan has been clicked on yet
        # do nothing
        if not self.is_initialized:
            return

        # if a line was not grabbed; do nothing
        if self.line_held is None:
            return

        # if not released in the axis; do nothing
        if not event.inaxes:
            return

        # convert xdata location to frequency index
        index = int(round(event.xdata*(len(self.data.freq)-1) / self.data.freq[-1], 0))

        # for the index to be valid to index the frequency array
        if index < 0:
            index = 0
        elif index > len(self.data.freq) - 1:
            index = len(self.data.freq) - 1

        # redraw the plot with the new line at given frequency
        self.plot(f=index)

        # also update cost plot with new frequency
        self.holder.cost_frame.plot(f=index)

        # reset line held to be None
        self.line_held = None
