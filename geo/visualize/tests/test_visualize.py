from geo.visualize import to_rgb
from numpy.testing import assert_equal
from geo.utils.testing import generate_test_dataset
import numpy as np


def test_to_rgb_gray():
    N = 4
    values = np.arange(N**2).reshape((N, N))
    rgb = np.repeat(values[:, :, np.newaxis], 3, axis=2) * 255 / (N**2 - 1)
    assert_equal(
        to_rgb(values, pmin=0, pmax=100), rgb
    )


def test_to_rgb_color():
    N = 2
    values = [
        np.arange(N**2).reshape((N, N)),
        np.ones((N, N)),
        np.ones((N, N))
    ]
    rgb = np.stack(
        [v if v.max() == v.min() else (v - v.min()) / (v.max() - v.min()) * 255
         for v in values], axis=-1
    )
    assert_equal(
        to_rgb(values, pmin=0, pmax=100), rgb
    )
