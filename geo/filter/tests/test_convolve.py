from geo.filter import _convolve, convolve, boxcar
from geo.testing import generate_test_dataset
from geo.io import assemble_complex
import numpy as np
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical


ds = generate_test_dataset()
identity_kernel = np.zeros((3, 3))
identity_kernel[1, 1] = 1


def test_convolve_ndarray():
    np.random.seed(42)
    arr = np.random.rand(20, 20)
    convolved = _convolve(arr, identity_kernel)
    assert_equal(arr, convolved)


def test_convolve_dataset_identity():
    convolved = convolve(ds, identity_kernel)
    xr_assert_identical(ds, convolved)


def test_convolve_dataset():
    np.random.seed(42)
    kernel = np.random.rand(5, 5)
    assert_equal(
        convolve(ds, kernel).C11.values,
        _convolve(ds.C11.values, kernel)
    )


def test_convolve_complex():
    ds_complex = assemble_complex(ds)
    convolved_complex = convolve(ds_complex, identity_kernel)
    xr_assert_identical(
        ds_complex, convolved_complex
    )


def test_boxcar():
    w = 5
    kernel = np.ones((w, w)) / w**2
    xr_assert_identical(
        boxcar(ds, w), convolve(ds, kernel)
    )
