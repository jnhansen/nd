from nd.filters import convolve, boxcar
from nd.filters.convolve_ import _convolve, _expand_kernel
from nd.testing import generate_test_dataset
from nd.io import assemble_complex
import numpy as np
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical


ds = generate_test_dataset()
identity_kernel = np.zeros((3, 3))
identity_kernel[1, 1] = 1


def test_expand_kernel():
    kernel = np.ones((2, 3))
    dims = ('x', 'y')
    new_dims = ('x', 'a', 'y', 's')
    new_kernel = _expand_kernel(kernel, dims, new_dims)
    assert_equal(
        new_kernel.shape, (2, 1, 3, 1)
    )


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
    dims = ('lat', 'lon')
    nd_kernel = _expand_kernel(kernel, dims, ds.C11.dims)
    assert_equal(
        convolve(ds, kernel, dims).C11.values,
        _convolve(ds.C11.values, nd_kernel)
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
