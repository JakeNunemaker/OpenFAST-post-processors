__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import numpy as np
import pandas as pd
from scipy import stats


def dataproperty(f):
    @property
    def wrapper(self, *args, **kwargs):
        if getattr(self, "data", None) is None:
            raise AttributeError("Output has not been read yet.")
        return f(self, *args, **kwargs)

    return wrapper


class OpenFASTOutput:
    """Base OpenFAST output class."""

    def __str__(self):
        return self.description

    @property
    def description(self):
        return getattr(self, "_desc", f"Unread OpenFAST output at '{self.filepath}'")

    @dataproperty
    def df(self):
        """Returns `self.data` as a DataFrame."""

        return pd.DataFrame(self.data, columns=self.headers)

    @dataproperty
    def num_timesteps(self):
        return self.data.shape[0]

    @dataproperty
    def num_channels(self):
        return self.data.shape[1]

    @dataproperty
    def idxmins(self):
        return np.argmin(self.data, axis=0)

    @dataproperty
    def idxmaxs(self):
        return np.argmax(self.data, axis=0)

    @dataproperty
    def minima(self):
        return np.min(self.data, axis=0)

    @dataproperty
    def maxima(self):
        return np.max(self.data, axis=0)

    @dataproperty
    def ranges(self):
        return self.maxima - self.minima

    @dataproperty
    def variable(self):
        return np.where(self.ranges != 0.0)[0]

    @dataproperty
    def constant(self):
        return np.where(self.ranges == 0.0)[0]

    @dataproperty
    def sums(self):
        return np.sum(self.data, axis=0)

    @dataproperty
    def sums_squared(self):
        return np.sum(self.data ** 2, axis=0)

    @dataproperty
    def sums_cubed(self):
        return np.sum(self.data ** 3, axis=0)

    @dataproperty
    def sums_fourth(self):
        return np.sum(self.data ** 4, axis=0)

    @dataproperty
    def sums_fourth(self):
        return np.sum(self.data ** 4, axis=0)

    @dataproperty
    def second_moments(self):
        return stats.moment(self.data, moment=2, axis=0)

    @dataproperty
    def third_moments(self):
        return stats.moment(self.data, moment=3, axis=0)

    @dataproperty
    def fourth_moments(self):
        return stats.moment(self.data, moment=4, axis=0)

    @dataproperty
    def means(self):
        means = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        means[:, self.constant] = self.minima[self.constant]
        means[:, self.variable] = self.sums / self.num_timesteps
        return means

    @dataproperty
    def stddevs(self):
        stddevs = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        stddevs[:, self.variable] = np.sqrt(self.second_moments)[self.variable]
        return stddevs

    @dataproperty
    def skews(self):
        skews = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        skews[:, self.variable] = (
            self.third_moments[self.variable] / self.second_moments[self.variable] ** 3
        )
        return skews

    @dataproperty
    def kurtosis(self):
        kurtosis = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        kurtosis[:, self.variable] = (
            self.fourth_moments[self.variable]
        ) / self.second_moments[self.variable] ** 2
        return kurtosis


class OpenFASTBinary(OpenFASTOutput):
    """OpenFAST binary output class."""

    def __init__(self, filepath, **kwargs):
        """
        Creates an instance of `OpenFASTBinary`.

        Parameters
        ----------
        filepath : path-like
        """

        self.filepath = filepath
        self._chan_chars = kwargs.get("chan_char_length", 10)
        self._unit_chars = kwargs.get("unit_char_length", 10)

    def read(self):
        """Reads the binary file."""

        with open(self.filepath, "rb") as f:
            self.fmt = np.fromfile(f, np.int16, 1)[0]

            num_channels = np.fromfile(f, np.int32, 1)[0]
            num_timesteps = np.fromfile(f, np.int32, 1)[0]
            num_points = num_channels * num_timesteps
            time_info = np.fromfile(f, np.float64, 2)
            self.slopes = np.fromfile(f, np.float32, num_channels)
            self.offset = np.fromfile(f, np.float32, num_channels)

            length = np.fromfile(f, np.int32, 1)[0]
            chars = np.fromfile(f, np.uint8, length)
            self._desc = "".join(map(chr, chars)).strip()

            self.build_headers(f, num_channels)
            time = self.build_time(f, time_info, num_timesteps)

            raw = np.fromfile(f, np.int16, count=num_points).reshape(
                num_timesteps, num_channels
            )
            self.data = np.concatenate(
                [time.reshape(num_timesteps, 1), (raw - self.offset) / self.slopes], 1
            )

    def build_headers(self, f, num_channels):
        """
        Builds the channels, units and headers arrays.

        Parameters
        ----------
        f : file
        num_channels : int
        """

        channels = np.fromfile(
            f, np.uint8, self._chan_chars * (num_channels + 1)
        ).reshape((num_channels + 1), self._chan_chars)
        self.channels = np.array(list("".join(map(chr, c)).strip() for c in channels))

        units = np.fromfile(f, np.uint8, self._unit_chars * (num_channels + 1)).reshape(
            (num_channels + 1), self._unit_chars
        )
        self.units = np.array(list("".join(map(chr, c)).strip()[1:-1] for c in units))

        self.headers = [
            self.channels[k]
            if self.units[k] in ["", "-"]
            else f"{self.channels[k]} [{self.units[k]}]"
            for k in range(num_channels + 1)
        ]

    def build_time(self, f, info, length):
        """
        Builds the time index based on the file format and index info.

        Parameters
        ----------
        f : file
        info : tuple
            Time index meta information, ('scale', 'offset') or ('t1', 'incr').

        Returns
        -------
        np.ndarray
            Time index for `self.data`.
        """

        if self.fmt == 1:
            scale, offset = info
            data = np.fromfile(f, np.int32, length)
            return (data - offset) / scale

        else:
            t1, incr = info
            return t1 + incr * np.arange(length)


class OpenFASTAscii(OpenFASTOutput):
    pass
