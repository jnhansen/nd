import pytest
import numpy as np
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from nd.testing import (generate_test_dataset, generate_test_dataarray,
                        assert_equal_files)
from nd import warp, filters
from nd import to_rgb, write_video


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_reproject(generator):
    ds = generator()
    kwargs = dict(crs='epsg:27700')
    xr_assert_equal(
        warp.reproject(ds, **kwargs),
        ds.nd.reproject(**kwargs)
    )


def test_accessor_nd_to_rgb():
    ds = generate_test_dataset(dims={'y': 50, 'x': 50})

    def rgb(d):
        return [d.C11, d.C22, d.C11/d.C22]

    assert_equal(
        to_rgb(rgb(ds)),
        ds.nd.to_rgb(rgb)
    )


def test_accessor_nd_to_video(tmpdir):
    ds = generate_test_dataset()

    path_1 = str(tmpdir.join('video1.avi'))
    path_2 = str(tmpdir.join('video2.avi'))

    write_video(ds, path_1)
    ds.nd.to_video(path_2)

    assert_equal_files(path_1, path_2)


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
