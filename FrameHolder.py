import pdb

from IndexFrame import IndexFrame
from WaveformFrame import WaveformFrame
from NScanFrame import NScanFrame
from CostFrame import CostPlot
from MagPhasePlot import MagPhasePlot


class FrameHolder:
    """
    Class to hold frame instances
    """

    def __init__(self, data, n_data, n_data_fmin, cost, e0, d):
        """
        Constructor for FrameHolder
        :param n_data: The 3D n_array that contains the solution for the index
            of refraction as a function of frequency
        """

        self.n = n_data
        self.cost = cost
        self.nfmin = n_data_fmin
        self.d = d

        start = 0
        while n_data[0, 0, start] == 0:
            start += 1

        self.start_index = start

        stop_index = 76

        # remove all frequency information after 2.5 THz
        data.freq = data.freq[:stop_index]
        data.freq_waveform = data.freq_waveform[:, :, :stop_index]

        # extent of the cost plot over (nr, ni) values
        extent = (-0.001, -1.0, 1, 2.5)

        self.n_frame = IndexFrame(self)
        self.a_scan_frame = WaveformFrame(self, data)
        self.n_scan_frame = NScanFrame(self, data)
        self.cost_frame = CostPlot(self, data, extent)
        self.mag_phase_frame = MagPhasePlot(self, data, e0[:stop_index])


class FrameHolder2:
    """
    Class to hold plots together and allow plots to interact with each other
    """

    def __init__(self, n_data, waveform, freq_waveform, time, freq, peak_bin):

        self.n = n_data
        self.waveform = waveform
        self.freq_waveform = freq_waveform
        self.time = time
        self.freq = freq
        self.peak_bin = peak_bin

        start = 0
        while n_data[0, 0, start] == 0:
            start += 1

        self.start_index = start

        self.n_plot = IndexPlot(self)
        # self.a_scan = WaveformFrame2(self)
        # self.n_scan = NScanPlot2(self)