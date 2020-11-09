__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os
from fnmatch import fnmatch

import numpy as np
import pandas as pd
import fatpack
from scipy.stats import weibull_min

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
        operating_files : list (optional)
            Operating files to read.
            Default: []
        idling_files : list (optional)
            Idling files to read.
            Default: []
        discrete_files : list (optional)
            Discrete files to read.
            Default: []
        aggregate_statistics : bool (optional)
            Flag for calculating aggregate statistics.
            Default: True
        calculated_channels : list (optional)
            Flags for additional calculated channels.
            Default: []
        fatigue_channels : dict (optional)
            Dictionary with format:
            'channel': 'fatigue slope'
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
        self.initialize(**kwargs)

    @staticmethod
    def valid_extension(fp, extensions):
        return any([fnmatch(fp, ext) for ext in extensions])

    def parse_settings(self, **kwargs):
        """Parses settings from input kwargs."""

        self._files = {
            "operating": kwargs.get("operating_files", []),
            "idle": kwargs.get("idling_files", []),
            "discrete": kwargs.get("discrete_files", []),
        }

        self._cc = kwargs.get("calculated_channels", [])
        self._fc = kwargs.get("fatigue_channels", {})
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

    def initialize(self, **kwargs):
        """Initializes required data structures."""

        self._summary_stats = {}
        self._aggregate_stats = {}
        self._dels = []
        self._samples = 0
        self._channels = np.ndarray(shape=(0,))
        self._elapsed = {}
        self._minima = np.ndarray(shape=(0,))
        self._maxima = np.ndarray(shape=(0,))
        self._ranges = np.ndarray(shape=(0,))
        self._sums = np.ndarray(shape=(0,))
        self._sums_squared = np.ndarray(shape=(0,))
        self._sums_cubed = np.ndarray(shape=(0,))
        self._sums_fourth = np.ndarray(shape=(0,))

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

    def process_files(self, **kwargs):
        """
        Processes all files for summary statistics, aggregate statistics and
        configured damage equivalent loads. This class runs all files in serial.
        """

        for i, f in enumerate(self.files):

            output = self.read_file(f)
            if i == 0:
                self._channels = output.channels
                self.initialize_sum_arrays(len(self.channels))

            else:
                if output.num_channels != len(self.channels):
                    raise ValueError("Channel mismatch detected.")

            self.append_summary_stats(output, **kwargs)
            self.append_aggregate_stats(output, **kwargs)
            self.append_DELs(output, **kwargs)

        self.post_process(**kwargs)

    def append_summary_stats(self, output, **kwargs):
        """
        Appends summary statistics to `self._summary_statistics` for each file.

        Parameters
        ----------
        output : OpenFASTOutput
        """

        fstats = {}
        for channel in output.channels:
            if channel in ["time", "Time"]:
                continue

            fstats[channel] = {
                "min": float(min(output[channel])),
                "max": float(max(output[channel])),
                "std": float(np.std(output[channel])),
                "mean": float(np.mean(output[channel])),
                "abs": float(max(np.abs(output[channel]))),
                "integrated": float(np.trapz(output["Time"], output[channel])),
            }

        self._summary_stats[output.filename] = fstats

    def append_aggregate_stats(self, output, **kwargs):
        """
        Reads `self.files`, appending the sums, sums^2, sums^3 and sums^4 to
        internal attributes for later use in `self.aggregate_stats`.

        Parameters
        ----------
        output : OpenFASTOutput
        """

        self._samples += output.num_timesteps
        self._elapsed[output.filename] = output.elapsed_time
        self.find_new_minima(output.minima)
        self.find_new_maxima(output.maxima)
        self.sums += output.sums
        self.sums_squared += output.sums_squared
        self.sums_cubed += output.sums_cubed
        self.sums_fourth += output.sums_fourth

    def post_process(self, **kwargs):
        """Post processes internal data to produce DataFrame outputs."""

        # Summary statistics
        ss = (
            pd.DataFrame.from_dict(self._summary_stats, orient="index")
            .stack()
            .to_frame()
        )
        ss = pd.DataFrame(ss[0].values.tolist(), index=ss.index)
        self._summary_stats = ss

        # Aggregate statistics
        agg = {}
        for idx, chan in enumerate(self.channels):
            if chan in ["time", "Time"]:
                continue

            agg[chan] = {
                "min": self.minima[idx],
                "max": self.maxima[idx],
                "mean": self.means[idx],
                "std": self.stddevs[idx],
                "skew": self.skews[idx],
                "kurtosis": self.kurtosis[idx],
            }
        self._aggregate_stats = pd.DataFrame(agg).T

        # Damage equivalent loads
        dels = pd.DataFrame(np.transpose(self._dels)).T
        dels.columns = self._fc.keys()
        dels["name"] = self.files
        self._dels = dels.set_index("name")

    def read_file(self, f):
        """
        Reads input file `f` and returns an instsance of one of the
        `OpenFASTOutput` subclasses.

        Parameters
        ----------
        f : str
            Filename that is appended to `self.directory`
        """

        fp = os.path.join(self.directory, f)
        if f.endswith(
            "outb"
        ):  # TODO: Convert to try/except with UnicodeError?
            output = OpenFASTBinary(fp, calculated_channels=self._cc)
            output.read()

        elif f.endswith("out"):
            output = OpenFASTAscii(fp, calculated_channels=self._cc)
            output.read()

        else:
            raise NotImplementedError("Other file formats not supported yet.")

        return output

    def get_load_rankings(self, ranking_vars, ranking_stats, **kwargs):
        """
        Returns load rankings across all files in `self.files`.

        Parameters
        ----------
        rankings_vars : list
            List of variables to evaluate for the ranking process.
        ranking_stats : list
            Summary statistic to evalulate. Currently supports 'min', 'max',
            'abs', 'mean', 'std'.
        """

        # TODO: Option to exclude operating/idling/discrete files.
        out = []
        for var, stat in zip(ranking_vars, ranking_stats):

            if not isinstance(var, list):
                var = list(var)

            col = pd.MultiIndex.from_product([self.files, var])

            if stat in ["max", "abs"]:
                res = (
                    *self.summary_stats.loc[col][stat].idxmax(),
                    stat,
                    self.summary_stats.loc[col][stat].max(),
                )

            elif stat == "min":
                res = (
                    *self.summary_stats.loc[col][stat].idxmin(),
                    stat,
                    self.summary_stats.loc[col][stat].min(),
                )

            elif stat in ["mean", "std"]:
                res = (
                    np.NaN,
                    ", ".join(var),
                    stat,
                    self.summary_stats.loc[col][stat].mean(),
                )

            else:
                raise NotImplementedError(
                    f"Statistic '{stat}' not supported for load ranking."
                )

            out.append(res)

        return pd.DataFrame(out, columns=["file", "channel", "stat", "val"])

    def find_new_minima(self, new):
        """Returns new minima across `self._minima` and `new`."""
        if self._minima.size == 0:
            self._minima = new
            return

        self._minima = np.min(
            np.append(self._minima, new).reshape(2, len(new)), axis=0
        )

    def find_new_maxima(self, new):
        """Returns new maxima across `self._maxima` and `new`."""
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
    def summary_stats(self):
        """Returns summary statistics for all files in `self.files`."""

        if isinstance(self._summary_stats, dict):
            raise ValueError("Files have not been processed.")

        return self._summary_stats

    @property
    def aggregate_stats(self):
        """Returns aggregate statistics for all files in `self.files`."""

        if isinstance(self._aggregate_stats, dict):
            raise ValueError("Files have not been processed.")

        return self._aggregate_stats

    @property
    def DELs(self):
        """Returns damage equivalent loads for all channels in `self._fc`"""

        if isinstance(self._dels, list):
            raise ValueError("Files have not been processed.")

        return self._dels

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

    def append_DELs(self, output, **kwargs):
        """
        Appends computed damage equivalent loads for fatigue channels in
        `self._fc`.

        Parameters
        ----------
        output : OpenFASTOutput
        """

        DELs = []
        for chan, slope in self._fc.items():
            try:
                DEL = self._compute_del(
                    output[chan], slope, output.elapsed_time, **kwargs
                )
                DELs.append(DEL)

            except IndexError as e:
                print(f"Channel '{chan}' not found for DEL calculation.")
                DELS.append(np.NaN)

        self._dels.append(DELs)

    @staticmethod
    def _compute_del(ts, slope, elapsed, **kwargs):
        """
        Computes damage equivalent load of input `ts`.

        Parameters
        ----------
        ts : np.array
            Time series to calculate DEL for.
        slope : int | float
            Slope of the fatigue curve.
        elapsed : int | float
            Elapsed time of the time series.
        rainflow_bins : int
            Number of bins used in rainflow analysis.
            Default: 100
        """

        bins = kwargs.get("rainflow_bins", 100)

        ranges = fatpack.find_rainflow_ranges(ts)
        Nrf, Srf = fatpack.find_range_count(ranges, 100)
        DELs = Srf ** slope * Nrf / elapsed
        DEL = DELs.sum() ** (1 / slope)

        return DEL
