__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os
from fnmatch import fnmatch

import numpy as np
from scipy.signal import find_peaks

from OpenFAST_IO import OpenFASTBinary


class pyLife:
    """Implementation of `mlife` in python."""

    def __init__(
        self,
        directory,
        extensions=["*.outb"],
        aggregate_statistics=True,
        fatigue_channels=[],
        filter_threshold=0,
    ):
        """
        Creates an instance of `pyLife`.

        Parameters
        ----------
        directory : path-like
            Path to OpenFAST output files.
        extensions : list
            List of file extensions to consider.
            Default: ['*.out', '*.outb']
        aggregate_statistics : bool
            Flag for calculating aggregate statistics.
        fatigue_channels : list
            List of channels to perform fatigue analysis on.
            Default: []
        filter_threshold : int | lit
            Threshold to apply to peak finding algorithm.
            Default: 0.
        """

        # Settings and file information
        self.directory = directory
        self._files = [
            fn for fn in os.listdir(directory) if self.valid_extension(fn, extensions)
        ]
        self._as = aggregate_statistics
        self._fc = fatigue_channels
        self._ft = filter_threshold

        # Initialize data structures
        self._samples = 0
        self._channels = np.ndarray(shape=(0,))
        self._elapsed = {}
        self._peaks = {}
        self._minima = np.ndarray(shape=(0,))
        self._maxima = np.ndarray(shape=(0,))
        self._ranges = np.ndarray(shape=(0,))
        self._sums = np.ndarray(shape=(0,))
        self._sums_squared = np.ndarray(shape=(0,))
        self._sums_cubed = np.ndarray(shape=(0,))
        self._sums_fourth = np.ndarray(shape=(0,))

    @staticmethod
    def valid_extension(fp, extensions):
        return any([fnmatch(fp, ext) for ext in extensions])

    def read_files(self):
        """
        Reads `self.files`, appending the sums, sums^2, sums^3 and sums^4 to
        internal attributes.
        """

        for i, f in enumerate(self.files):

            fp = os.path.join(self.directory, f)
            if f.endswith("outb"):
                output = OpenFASTBinary(fp)
                output.read()

            elif f.endswith("out"):
                raise NotImplemented("ASCII input not yet implemented.")

            if i == 0:
                self._channels = output.channels
                self.initialize_sum_arrays(len(self.channels))

            else:
                if output.num_channels != len(self.channels):
                    raise ValueError("Channel mismatch detected.")

            self._samples += output.num_timesteps
            self._elapsed[f] = output.elapsed_time

            if self._as:
                self.find_new_minima(output.minima)
                self.find_new_maxima(output.maxima)
                self.sums += output.sums
                self.sums_squared += output.sums_squared
                self.sums_cubed += output.sums_cubed
                self.sums_fourth += output.sums_fourth

            if self._fc:
                file_peaks = {}
                for chan in self._fc:
                    idx = np.where(self.channels == chan)[0]
                    file_peaks[chan] = self.determine_peaks(
                        output.data[:, idx], prominence=self._ft
                    )

                self._peaks[f] = file_peaks

    def initialize_sum_arrays(self, num_chan):
        """
        Initializes arrays for sums, sums^2, sums^3, sums^4.

        Parameters
        ----------
        num_channels : int
            Number of channels to aggregate sums on.
        """

        self._sums = np.zeros(shape=(1, num_chan), dtype=np.float64)
        self._sums_squared = np.zeros(shape=(1, num_chan), dtype=np.float64)
        self._sums_cubed = np.zeros(shape=(1, num_chan), dtype=np.float64)
        self._sums_fourth = np.zeros(shape=(1, num_chan), dtype=np.float64)

    def find_new_minima(self, new):
        if self._minima.size == 0:
            self._minima = new
            return

        self._minima = np.min(np.append(self._minima, new).reshape(2, len(new)), axis=0)

    def find_new_maxima(self, new):
        if self._maxima.size == 0:
            self._maxima = new
            return

        self._maxima = np.max(np.append(self._maxima, new).reshape(2, len(new)), axis=0)

    @property
    def files(self):
        return self._files

    @property
    def filepaths(self):
        return [os.path.join(self.directory, fn) for fn in self._files]

    @property
    def samples(self):
        if self._samples == 0:
            raise ValueError("No files have been read.")

        return self._samples

    @property
    def channels(self):
        return self._channels

    @property
    def elapsed_times(self):
        return self._elapsed

    @property
    def maxima(self):
        return self._maxima

    @property
    def minima(self):
        return self._minima

    @property
    def ranges(self):
        return self._maxima - self._minima

    @property
    def variable(self):
        return np.where(self.ranges != 0.0)[0]

    @property
    def constant(self):
        return np.where(self.ranges == 0.0)[0]

    @property
    def sums(self):
        return self._sums

    @sums.setter
    def sums(self, new):
        self._sums = new

    @property
    def means(self):
        return self.sums / self.samples

    @property
    def sums_squared(self):
        return self._sums_squared

    @sums_squared.setter
    def sums_squared(self, new):
        self._sums_squared = new

    @property
    def sums_cubed(self):
        return self._sums_cubed

    @sums_cubed.setter
    def sums_cubed(self, new):
        self._sums_cubed = new

    @property
    def sums_fourth(self):
        return self._sums_fourth

    @sums_fourth.setter
    def sums_fourth(self, new):
        self._sums_fourth = new

    @property
    def second_moments(self):
        return (
            self.sums_squared
            - (2 * self.means * self.sums)
            + self.samples * self.means ** 2
        ) / self.samples

    @property
    def third_moments(self):
        return (
            self.sums_cubed
            - (3 * self.means * self.sums_squared)
            + (3 * (self.means ** 2) * self.sums)
            - self.samples * self.means ** 3
        ) / self.samples

    @property
    def fourth_moments(self):
        return (
            self.sums_fourth
            - (4 * self.sums_cubed * self.means)
            + (6 * self.sums_squared * (self.means ** 2))
            - (4 * self.sums * (self.means ** 3))
            + self.samples * self.means ** 4
        ) / self.samples

    @property
    def stddevs(self):
        return np.sqrt(self.second_moments)

    @property
    def skews(self):
        skews = np.zeros(self.sums.shape, dtype=np.float64)
        skews[:, self.variable] = (
            self.third_moments[:, self.variable]
            / np.sqrt(self.second_moments[:, self.variable]) ** 3
        )
        return skews

    @property
    def kurtosis(self):
        kurtosis = np.zeros(self.sums.shape, dtype=np.float64)
        kurtosis[:, self.variable] = (
            self.fourth_moments[:, self.variable]
            / self.second_moments[:, self.variable] ** 2
        )
        return kurtosis

    def determine_peaks(self, data, prominence):
        """
        Finds the inflection points of `data` with required `prominence`.

        Parameters
        ----------
        data : np.array
        prominence : int | float
            Required prominence to be considered a peak.

        Returns
        -------
        np.array
            Array of filtered peaks in `data`.
        """

        infl = self.find_extrema(data)
        _max, _ = find_peaks(infl, prominence=prominence)
        _min, _ = find_peaks(-infl, prominence=prominence)
        idx = np.array([0, *np.sort(np.append(_max, _min)), len(infl) - 1])

        return infl[idx]

    @staticmethod
    def find_extrema(data):
        """
        Implementation of `mlife.determine_peaks`.

        Parameters
        ----------
        data : np.array

        Returns
        -------
        np.array
            Array of inflection points in `data`.
        """

        end = data[-1]
        data = data[np.where((data[1:] - data[:-1] != 0))[0]]
        data = np.append(data, end)

        back = data[1:-1] - data[:-2]
        forw = data[2:] - data[1:-1]
        sign = np.sign(back) + np.sign(forw)
        idx = np.unique([0, *np.where(sign == 0)[0] + 1, len(data) - 1])

        return data[idx]
