__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import numpy as np
import pytest

from OpenFAST_Processors import pyLife


def test_directory_read(data_dir):

    analysis = pyLife(data_dir, operating_files=[f"IEA15MW_DLC_ED_{i:03d}.outb" for i in range(5)])
    assert len(analysis.files) == 5

    analysis.compute_aggregate_statistics()
    assert analysis.samples == 5 * 60001


def test_pylife_vs_combined_dataset(data_dir, combined_output):

    life = pyLife(data_dir, operating_files=[f"IEA15MW_DLC_ED_{i:03d}.outb" for i in range(5)])
    life.compute_aggregate_statistics()

    # Number of samples
    assert life.samples == combined_output.num_timesteps

    # Intermediate results
    assert np.allclose(life.sums, combined_output.sums)
    assert np.allclose(life.sums_squared, combined_output.sums_squared)
    assert np.allclose(life.sums_cubed, combined_output.sums_cubed)
    assert np.allclose(life.sums_fourth, combined_output.sums_fourth)

    # Minima/maxima
    assert np.all(life.minima == combined_output.minima)
    assert np.all(life.maxima == combined_output.maxima)

    # Moments
    assert np.allclose(life.second_moments, combined_output.second_moments)
    assert np.allclose(
        life.third_moments[:, 1:], combined_output.third_moments[1:], rtol=1e-4
    )
    assert np.allclose(life.fourth_moments, combined_output.fourth_moments, rtol=1e-3)

    # Stats
    assert np.allclose(life.means, combined_output.means)
    assert np.allclose(life.skews, combined_output.skews, rtol=1e-4)
    assert np.allclose(life.kurtosis, combined_output.kurtosis, rtol=1e-3)
