import pytest
import numpy as np
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical
from nd.testing import generate_test_dataset, generate_test_dataarray
from nd import warp, filters


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_reproject(generator):
    ds = generator()
    kwargs = dict(crs='epsg:27700')
    xr_assert_equal(
        warp.reproject(ds, **kwargs),
        ds.nd.reproject(**kwargs)
    )


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_filter_nlmeans(generator):
    ds = generator()
    kwargs = dict(dims=('y', 'x'), r=0, f=1, sigma=2, h=2)
    xr_assert_equal(
        filters.nlmeans(ds, **kwargs),
        ds.filter.nlmeans(**kwargs)
    )


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_filter_boxcar(generator):
    ds = generator()
    kwargs = dict(dims=('y', 'x'), w=5)
    xr_assert_equal(
        filters.boxcar(ds, **kwargs),
        ds.filter.boxcar(**kwargs)
    )


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_filter_convolution(generator):
    ds = generator()
    np.random.seed(42)
    kwargs = dict(dims=('y', 'x'), kernel=np.random.rand(5, 5))
    xr_assert_equal(
        filters.convolution(ds, **kwargs),
        ds.filter.convolve(**kwargs)
    )


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_filter_gaussian(generator):
    ds = generator()
    kwargs = dict(dims=('y', 'x'), sigma=1.5)
    xr_assert_equal(
        filters.gaussian(ds, **kwargs),
        ds.filter.gaussian(**kwargs)
    )
