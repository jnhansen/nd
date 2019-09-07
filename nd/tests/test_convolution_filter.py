from nd.filters import ConvolutionFilter, BoxcarFilter, _expand_kernel
from nd.testing import generate_test_dataset
from nd.io import assemble_complex
import scipy.ndimage.filters as snf
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
    convolved = snf.convolve(arr, identity_kernel)
    assert_equal(arr, convolved)


def test_convolve_dataset_identity():
    dims = ('y', 'x')
    convolved = ConvolutionFilter(dims, identity_kernel).apply(ds)
    xr_assert_identical(ds, convolved)


def test_convolve_dataset():
    np.random.seed(42)
    kernel = np.random.rand(5, 5)
    dims = ('y', 'x')
    nd_kernel = _expand_kernel(kernel, dims, ds.C11.dims)
    assert_equal(
        ConvolutionFilter(dims, kernel).apply(ds).C11.values,
        snf.convolve(ds.C11.values, nd_kernel)
    )


def test_convolve_complex():
    ds_complex = assemble_complex(ds)
    convolved_complex = ConvolutionFilter(
        ('y', 'x'), identity_kernel).apply(ds_complex)
    xr_assert_identical(
        ds_complex, convolved_complex
    )


def test_boxcar():
    w = 5
    dims = ('y', 'x')
    kernel = np.ones((w, w)) / w**2
    xr_assert_identical(
        BoxcarFilter(dims, w).apply(ds),
        ConvolutionFilter(dims, kernel).apply(ds)
    )
