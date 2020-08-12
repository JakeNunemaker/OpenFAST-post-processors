__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os
from fnmatch import fnmatch

import numpy as np
from scipy.stats import weibull_min
from scipy.signal import find_peaks

from OpenFAST_IO import OpenFASTAscii, OpenFASTBinary


class pyLife:
    """Implementation of `mlife` in python."""

    def __init__(self, directory, **kwargs):
        """
        Creates an instance of `pyLife`.

        Parameters
        ----------
        directory : path-like
            Path to OpenFAST output files.
        extensions : list
            List of extensions to include from `directory`.
            Not used if `files` is passed.
            Default: ["*.out", "*.outb"]
        files : list (optional)
            Files to read. Extensions must match `extensions`. If empty, find
            all files in `directory`.
            Default: []
        aggregate_statistics : bool (optional)
            Flag for calculating aggregate statistics.
            Default: True
        calculated_channels : list (optional)
            Flags for additional calculated channels.
            Default: []
        fatigue_channels : list (optional)
            List of channels to perform fatigue analysis on.
            Default: []
        filter_threshold : int | lit (optional)
            Threshold to apply to peak finding algorithm.
            Default: 0.
        weibull_shape : int | float (optional)
            Shape parameter of the windspeed distribution.
            Default: 2 [m/s]
        weibull_scale : int | float (optional)
            Scale parameter of the windspeed distribution.
            Default: 10 [m/s]
        ws_in : int | float (optional)
            Cut-in windspeed for the turbine.
            Default: 3 [m/s]
        ws_out : int | float (optional)
            Cut-out windspeed for the turbine.
            Default: 21 [m/s]
        ws_max : int | float (optional)
            Maximum windspeed value for bins.
            Default: 44 [m/s]
        max_bin_size : int | float (optional)
            Maximum width of a windspeed bin.
            Default: 1 [m/s]
        design_life : int | float (optional)
            Design lifetime of the turbine in seconds.
            Default: 630720000 [s]
        availability : int | float (optional)
            Frace of the design life that the turbine is operating between
            `ws_in` and `ws_out`.
            Default: 1
        uc_mult : float (optional)
            Multiplier for binning unclosed cycles. Discard: 0, Full Cycle: 1
            Default: 0.5
        goodman : bool (optional)
            Flag for using the Goodman corretion in the fatigue calculations.
            Default: True
        """

        self.directory = directory
        self.parse_settings(**kwargs)
        self.initialize_data_structures(**kwargs)

    def parse_settings(self, **kwargs):
        """Parses settings from input kwargs."""

        self._files = {
            "operating": kwargs.get("operating_files", []),
            "idle": kwargs.get("idling_files", []),
            "discrete": kwargs.get("discrete_files", [])
        }

        self._cc = kwargs.get("calculated_channels", [])
        self._fc = kwargs.get("fatigue_channels", [])
        self._ft = kwargs.get("filter_threshold", 0)
        self._shape = kwargs.get("weibull_shape", 2)
        self._scale = kwargs.get("weibull_scale", 10)
        self._vin = kwargs.get("ws_in", 3)
        self._vout = kwargs.get("ws_out", 21)
        self._vmax = kwargs.get("ws_max", 44)
        self._max_bin = kwargs.get("max_bin_size", 1)
        self._design_life = kwargs.get("design_life", 630720000)
        self._avail = kwargs.get("availability", 1)
        self._uc_mult = kwargs.get("uc_mult", 0.5)
        self._goodman = kwargs.get("goodman", True)

    def initialize_data_structures(self, **kwargs):
        """Initializes required data structures."""

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

    def compute_aggregate_statistics(self):
        """
        Reads `self.files`, appending the sums, sums^2, sums^3 and sums^4 to
        internal attributes.
        """

        for i, f in enumerate(self.files):

            fp = os.path.join(self.directory, f)
            if f.endswith("outb"):
                output = OpenFASTBinary(fp, calculated_channels=self._cc)
                output.read()

            elif f.endswith("out"):
                output = OpenFASTAscii(fp, calculated_channels=self._cc)
                output.read()

            if i == 0:
                self._channels = output.channels
                self.initialize_sum_arrays(len(self.channels))

            else:
                if output.num_channels != len(self.channels):
                    raise ValueError("Channel mismatch detected.")

            self._samples += output.num_timesteps
            self._elapsed[f] = output.elapsed_time
            self.find_new_minima(output.minima)
            self.find_new_maxima(output.maxima)
            self.sums += output.sums
            self.sums_squared += output.sums_squared
            self.sums_cubed += output.sums_cubed
            self.sums_fourth += output.sums_fourth
    
    def compute_fatigue(self):
        """
        TODO:
        """

        for i, f in enumerate(self.files):
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

        self._minima = np.min(
            np.append(self._minima, new).reshape(2, len(new)), axis=0
        )

    def find_new_maxima(self, new):
        if self._maxima.size == 0:
            self._maxima = new
            return

        self._maxima = np.max(
            np.append(self._maxima, new).reshape(2, len(new)), axis=0
        )

    @property
    def operating_files(self):
        return self._files["operating"]
    
    @property
    def idle_files(self):
        return self._files["idle"]
    
    @property
    def discrete_files(self):
        return self._files["discrete"]

    @property
    def files(self):
        return [*self.operating_files, *self.idle_files, *self.discrete_files]

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
        return (self.sums / self.samples).flatten()

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
        return np.sqrt(self.second_moments).flatten()

    @property
    def skews(self):
        skews = np.zeros(self.sums.shape, dtype=np.float64)
        skews[:, self.variable] = (
            self.third_moments[:, self.variable]
            / np.sqrt(self.second_moments[:, self.variable]) ** 3
        )
        return skews.flatten()

    @property
    def kurtosis(self):
        kurtosis = np.zeros(self.sums.shape, dtype=np.float64)
        kurtosis[:, self.variable] = (
            self.fourth_moments[:, self.variable]
            / self.second_moments[:, self.variable] ** 2
        )
        return kurtosis.flatten()

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

    def compute_windspeed_bins(self):
        """
        Finds bins of width less than or equual to `self._max_bin` for the
        following ranges:
        - 0 to cut-in
        - cut-in to cut-out
        - cut-out to max

        Returns
        -------
        np.array
            Upper boundaries of each bin.
        np.array
            Probabilities based on the CDF of the Weibull distribution defined
            by `self._shape` and `self._scale`.
        """

        args = (self._max_bin, self._shape, self._scale)
        _, _, upper1, probabilities1 = self.compute_bins(0, self._vin, *args)
        _, _, upper2, probabilities2 = self.compute_bins(
            self._vin, self._vout, *args
        )
        _, _, upper3, probabilities3 = self.compute_bins(
            self._vout, self._vmax, *args
        )

        boundaries = np.concatenate([upper1, upper2, upper3])
        probabilities = np.concatenate(
            [probabilities1, probabilities2, probabilities3]
        )

        return boundaries, probabilities

    @staticmethod
    def compute_bins(vmin, vmax, maxbin, c, k):
        """
        Finds bin boundaries with maximum `vmax`, minimum `vmin` and max bin
        size `maxbin`. Also returns the weibull cumulative distribution
        probability of each bin with shape factor `c` and scale factor `k`.

        Parameters
        ----------
        vmin : int | float
            Range minimum.
        vmax : int | float
            Range maximum.
        maxbin : int | float
            Maximum bin width.
        c : int | float
            Weibull shape factor.
        k : int | float
            Weibull scale factor.

        Returns
        -------
        int
            Number of bins.
        float
            Bin width.
        list
            List of upper boundaries of each bin.
        np.array
            Weibull probability of each bin.
        """

        num_bins = int(np.ceil((vmax - vmin) / maxbin))
        probabilities = np.zeros(num_bins)
        width = (vmax - vmin) / num_bins
        upper_bounds = []

        for i in range(num_bins):

            bot = vmin + i * width
            top = bot + width
            upper_bounds.append(top)

            dist = weibull_min(c, scale=k)
            probabilities[i] = dist.cdf(top) - dist.cdf(bot)

        return num_bins, width, upper_bounds, probabilities
