import pytest
import numpy as np
import inspect
import xarray as xr
from collections import OrderedDict
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from nd.testing import (generate_test_dataset, generate_test_dataarray,
                        assert_equal_files, assert_equal_crs, requires)
from nd import warp, filters, io, change, utils, visualize
from nd._xarray import patch_doc
from rasterio.crs import CRS
try:
    import cartopy
except ModuleNotFoundError:
    cartopy = None


# ---------------
# Test properties
# ---------------

@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_shape(generator):
    ds = generator()
    shape = utils.get_shape(ds)
    assert_equal(shape, ds.nd.shape)


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_dims(generator):
    ds = generator()
    dims = utils.get_dims(ds)
    assert_equal(dims, ds.nd.dims)


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_crs(generator):
    crs = CRS.from_epsg(4326)
    ds = generator(crs=crs)
    assert_equal_crs(crs, ds.nd.crs)


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_bounds(generator):
    extent = (10, 30, 15, 35)
    ds = generator(extent=extent)
    assert_equal(extent, ds.nd.bounds)


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_resolution(generator):
    ds = generator()
    res = warp.get_resolution(ds)
    assert_equal(res, ds.nd.resolution)


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_transform(generator):
    ds = generator()
    transf = warp.get_transform(ds)
    assert_equal(transf, ds.nd.transform)


# -------------------------
# Test reprojection methods
# -------------------------

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


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_resample(generator):
    ds = generator()
    kwargs = dict(width=50)
    xr_assert_equal(
        warp.resample(ds, **kwargs),
        ds.nd.resample(**kwargs)
    )


# --------------------------
# Test visualization methods
# --------------------------

def test_accessor_nd_to_rgb():
    ds = generate_test_dataset(dims={'y': 50, 'x': 50})

    def rgb(d):
        return [d.C11, d.C22, d.C11/d.C22]

    assert_equal(
        visualize.to_rgb(rgb(ds)),
        ds.nd.to_rgb(rgb)
    )


def test_accessor_nd_to_rgb_default():
    ds = generate_test_dataset(dims={'y': 50, 'x': 50})

    assert_equal(
        visualize.to_rgb([ds.C11, ds.C22, ds.C11/ds.C22]),
        ds.nd.to_rgb()
    )


def test_accessor_nd_to_video(tmpdir):
    ds = generate_test_dataset()

    path_1 = str(tmpdir.join('video1.avi'))
    path_2 = str(tmpdir.join('video2.avi'))

    visualize.write_video(ds, path_1)
    ds.nd.to_video(path_2)

    assert_equal_files(path_1, path_2)


@requires('cartopy')
def test_accessor_nd_plot_map():
    ds = generate_test_dataset()

    ax = ds.nd.plot_map(background=None)
    assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot)


# ---------------
# Test IO methods
# ---------------

def test_accessor_nd_as_complex():
    ds = generate_test_dataset()

    xr_assert_equal(
        io.assemble_complex(ds),
        ds.nd.as_complex()
    )


def test_accessor_nd_as_real():
    ds = generate_test_dataset().nd.as_complex()

    xr_assert_equal(
        io.disassemble_complex(ds),
        ds.nd.as_real()
    )


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_to_netcdf(tmpdir, generator):
    ds = generator()
    path_1 = str(tmpdir.join('ds1.nc'))
    path_2 = str(tmpdir.join('ds2.nc'))

    io.to_netcdf(ds, path_1)
    ds.nd.to_netcdf(path_2)

    xr_assert_equal(
        io.open_dataset(path_1),
        io.open_dataset(path_2)
    )


# --------------------
# Test general methods
# --------------------
@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_accessor_nd_apply(generator):
    ds = generator()

    def func(arr):
        """Reduce a two dimensional array to its mean."""
        return arr.mean()

    signature = '(x,y)->()'

    xr_assert_equal(
        ds.nd.apply(func, signature=signature),
        utils.apply(ds, func, signature=signature)
    )


# -------------------------------
# Test change detection accessors
# -------------------------------

@requires('gsl')
def test_accessor_nd_omnibus():
    ds1 = generate_test_dataset(
        dims={'y': 5, 'x': 5, 'time': 10},
        mean=[1, 0, 0, 1], sigma=0.1
        ).isel(time=slice(None, 5))
    ds2 = generate_test_dataset(
        dims={'y': 5, 'x': 5, 'time': 10},
        mean=[10, 0, 0, 10], sigma=0.1
        ).isel(time=slice(5, None))
    ds = xr.concat([ds1, ds2], dim='time')
    kwargs = dict(n=9, alpha=0.9)

    xr_assert_equal(
        change.omnibus(ds, **kwargs),
        ds.nd.change_omnibus(**kwargs)
    )


# ---------------------
# Test filter accessors
# ---------------------

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


# ---------------------------
# Test accessor documentation
# ---------------------------

def test_patch_doc():
    def src_fn(data, a, b, c, d={}):
        """Source docstring"""
        pass

    @patch_doc(src_fn)
    def fn(self):
        pass

    # Check that docstring matches
    assert_equal(src_fn.__doc__, fn.__doc__)

    # Check that signature matches
    # (apart from first parameter)
    params_src = OrderedDict(inspect.signature(src_fn).parameters)
    params_fn = OrderedDict(inspect.signature(fn).parameters)
    params_src.popitem(last=False)
    params_fn.popitem(last=False)
    assert_equal(
        params_src, params_fn
    )
