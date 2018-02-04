import pdb
import sys
import socket

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

# handles laptop vs desktop drive issue
if socket.gethostname() == 'Laptop':
    drive = 'C'
else:
    drive = 'D'

sys.path.insert(0, drive + ':\\PycharmProjects\\THzProcClass')

from ParentFrame import ParentFrame


class WaveformFrame(ParentFrame):
    """
    Displays the A-Scan of the point clicked and the FFT of the waveform
    between the follow gates
    """

    def __init__(self, holder, data):
        """
        Constructor method
        :param holder: Class that holds Frames together
        """

        # provide figure with two subplots stacked on top of each other
        super().__init__('A-Scan Frame', (2, 1))

        self.time_axis = self.axis[0]
        self.freq_axis = self.axis[1]
        del self.axis

        # The frame holder to hold all the plots together
        self.holder = holder

        # an instance of THzProcClass
        self.data = data

        self.is_initialized = False

        self.connect_events()

        plt.close()

    def plot(self, i, j):
        """
        Plots the time and frequency domain waveform from the given (i, j)
        location.
        """
        if not self.is_initialized:
            self.is_initialized = True

        self.time_axis.cla()
        self.freq_axis.cla()

        # plot time domain information
        self.time_axis.plot(self.data.time, self.data.waveform[i, j, :], 'r')

        # plot the gates
        left = self.data.peak_bin[3, 1, i, j]
        right = self.data.peak_bin[4, 1, i, j]
        self.time_axis.axvline(self.data.time[left], color='b')
        self.time_axis.axvline(self.data.time[right], color='g')

        # plot frequency waveform, at this point frequency waveform should
        # already contain information that is only between the plotted gates
        # in the time domain
        self.freq_axis.plot(self.data.freq, np.abs(self.data.freq_waveform[i, j, :]), 'r')

        self.time_axis.grid()
        self.freq_axis.grid()

        self.freq_axis.set_xlim(0, 3)

        self.time_axis.set_xlabel('Time (ps)')
        self.time_axis.set_ylabel('Amplitude')
        self.time_axis.set_title('(%d, %d)' % (i, j))

        self.freq_axis.set_xlabel('Frequency (THz)')
        self.freq_axis.set_ylabel('Amplitude')

        self.figure_canvas.draw()

    def connect_events(self):
        """
        Connects that matplotlib events to the appropriate method handler
        """
        self.figure_canvas.mpl_connect('motion_notify_event', self.motion_handler)

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

        string = '(%.4f, %.4f)' % (xid, yid)

        self.status_bar.SetStatusText(string)
