__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


# TODO: Tests for binary/ascii data reading with different OpenFAST formats.


import numpy as np
import pandas as pd
import pytest


def test_base_output(test_output):

    # Types
    assert isinstance(test_output.data, np.ndarray)
    assert isinstance(test_output.df, pd.DataFrame)

    # Sizes
    assert test_output.num_timesteps == 5
    assert test_output.num_channels == 5

    # Simple outputs
    assert np.all(test_output.minima == np.array([1, 0, 10, 1, 2]))
    assert np.all(test_output.maxima == np.array([5, 0, 10, 3, 8]))
    assert np.all(test_output.idxmins == np.array([0, 0, 0, 0, 0]))
    assert np.all(test_output.idxmaxs == np.array([4, 0, 0, 2, 3]))
    assert np.all(test_output.ranges == np.array([4, 0, 0, 2, 6]))

    # Constant/variable channels
    assert np.all(test_output.constant == np.array([1, 2]))
    assert np.all(test_output.variable == np.array([0, 3, 4]))

    # Intermediate outputs for stats
    assert np.all(test_output.sums == np.array([15, 0, 50, 9, 24]))
    assert np.all(test_output.sums_squared == np.array([55, 0, 500, 19, 136]))
    assert np.all(test_output.sums_cubed == np.array([225, 0, 5000, 45, 864]))
    assert np.all(test_output.sums_fourth == np.array([979, 0, 50000, 115, 5920]))

    # Statistics outputs
    assert test_output.means.shape == (5,)
    assert test_output.stddevs.shape == (5,)
    assert test_output.skews.shape == (5,)
    assert test_output.kurtosis.shape == (5,)


def test_sample_output(sample_output):

    assert sample_output.num_timesteps == 60001
    assert sample_output.num_channels == 88


def test_sample_output_unread(sample_output_unread):

    with pytest.raises(AttributeError):
        sample_output_unread.data
