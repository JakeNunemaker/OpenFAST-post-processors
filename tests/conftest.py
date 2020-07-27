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
def test_output():

    test_array = np.array(
        [[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [10, 10, 10, 10], [1, 2, 3, 2, 1]]
    )

    return OpenFASTOutput(test_array)


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
