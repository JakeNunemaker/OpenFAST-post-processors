"""Shared pytest settings and fixtures."""
__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os

import numpy as np
import pytest

from OpenFAST_IO import OpenFASTBinary, OpenFASTOutput

DIR = os.path.split(os.path.abspath(__file__))[0]


@pytest.fixture()
def data_dir():
    return os.path.join(DIR, "data")


@pytest.fixture()
def test_output():

    test_array = np.array(
        [
            [1, 0, 10, 1, 2],
            [2, 0, 10, 2, 4],
            [3, 0, 10, 3, 6],
            [4, 0, 10, 2, 8],
            [5, 0, 10, 1, 4],
        ]
    )

    return OpenFASTOutput(test_array, dlc="Test DLC")


@pytest.fixture()
def sample_output():

    filepath = os.path.join(DIR, "data", "IEA15MW_DLC_ED_000.outb")
    output = OpenFASTBinary(filepath)
    output.read()
    return output


@pytest.fixture()
def sample_output_unread():

    filepath = os.path.join(DIR, "data", "IEA15MW_DLC_ED_000.outb")
    output = OpenFASTBinary(filepath)
    return output


@pytest.fixture()
def combined_output():

    data = []
    for i in range(5):

        fp = os.path.join(DIR, "data", f"IEA15MW_DLC_ED_{i:03d}.outb")
        output = OpenFASTBinary(fp)
        output.read()

        data.append(output.data)

    return OpenFASTOutput(np.vstack(data), dlc="Combined DLC")
