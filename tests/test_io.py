__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import pytest


def test_base_output(test_output):

    assert test_output.data.size


def test_sample_output(sample_output):

    assert sample_output.data.size


def test_sample_output_unread(sample_output_unread):

    with pytest.raises(AttributeError):
        sample_output_unread.data
