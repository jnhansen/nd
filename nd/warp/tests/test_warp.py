import pytest
from nd.warp import (Reprojection, Resample, Alignment, get_bounds,
                     get_transform, get_crs, get_common_bounds,
                     get_common_extent, get_extent, get_resolution)
from nd.warp.warp_ import _parse_crs, nrows, ncols
from nd.io import open_dataset
from nd.testing import generate_test_dataset, assert_equal_crs
import numpy as np
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
import os
from rasterio.crs import CRS
from affine import Affine


# Prepare test data
ds_params = [
    ('notime', {
        # Default
        'nx': 10,
        'ny': 20,
        'ntime': None,
        'crs': CRS({'init': 'epsg:4326'})
    }),
    ('notime_mercator', {
        # Default
        'nx': 10,
        'ny': 20,
        'ntime': None,
        'crs': CRS({'init': 'epsg:3395'})
    }),
    ('standard', {
        # Default
        'nx': 10,
        'ny': 20,
        'ntime': 1,
        'crs': CRS({'init': 'epsg:4326'})
    }),
    ('standard_mercator', {
        # Test Mercator Projection
        'nx': 10,
        'ny': 20,
        'ntime': 1,
        'crs': CRS({'init': 'epsg:3395'})
    }),
    ('ntime=5', {
        # Test temporal dimension
        'nx': 10,
        'ny': 20,
        'ntime': 5,
        'crs': CRS({'init': 'epsg:4326'})
    }),
    ('variables', {
        # Test different variables
        'nx': 10,
        'ny': 20,
        'ntime': 1,
        'crs': CRS({'init': 'epsg:4326'}),
        'var': ['v1', 'v2', 'v3']
    })
]

data_path = 'data/'
nc_path = os.path.join(data_path, 'slc.nc')
tif_path = os.path.join(data_path, 'slc.tif')
dim_path = os.path.join(data_path, 'slc.dim')
slc_files = [nc_path, tif_path, dim_path]
slc_datasets = [open_dataset(f) for f in slc_files]


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_reprojection(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    crs = _parse_crs('+init=epsg:4326')
    proj = Reprojection(crs=crs)
    reprojected = proj.apply(ds)
    assert_equal_crs(crs, get_crs(reprojected))


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_resample_to_resolution_tuple(name, kwargs):
    res = (0.05, 0.01)
    ds = generate_test_dataset(**kwargs)
    resampled = Resample(res=res).apply(ds)
    assert_equal(res, get_resolution(resampled))


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_resample_to_resolution_float(name, kwargs):
    res = 0.05
    ds = generate_test_dataset(**kwargs)
    resampled = Resample(res=res).apply(ds)
    assert_equal((res, res), get_resolution(resampled))


@pytest.mark.parametrize('resample_kwargs', [{'width': 25}, {'height': 25}])
@pytest.mark.parametrize('name,kwargs', ds_params)
def test_resample_to_width_or_height(name, kwargs, resample_kwargs):
    ds = generate_test_dataset(**kwargs)
    resampled = Resample(**resample_kwargs).apply(ds)
    if 'width' in resample_kwargs:
        assert_equal(resample_kwargs['width'], ncols(resampled))
    elif 'height' in resample_kwargs:
        assert_equal(resample_kwargs['height'], nrows(resampled))

    # Make sure aspect ratio is preserved
    assert_equal(
        int(ncols(resampled) / nrows(resampled)),
        int(ncols(ds) / nrows(ds))
    )


def test_alignment():
    ...


@pytest.mark.parametrize('crs', [
    CRS.from_string('+init=epsg:4326')
])
def test_parse_crs(crs):
    assert_equal_crs(crs, _parse_crs(crs))
    assert_equal_crs(crs, _parse_crs(crs.to_string()))
    assert_equal_crs(crs, _parse_crs(crs.to_dict()))
    assert_equal_crs(crs, _parse_crs(crs.wkt))


@pytest.mark.skip(reason="This currently fails due to SNAP saving "
                         "inconsistent datasets.")
def test_equal_datasets():
    ds0 = slc_datasets[0]
    for ds in slc_datasets[1:]:
        assert_equal(ds0['x'].values, ds['x'].values,
                     'x coordinates are not equal')
        assert_equal(ds0['y'].values, ds['y'].values,
                     'y coordinates are not equal')
        assert_equal(get_transform(ds0), get_transform(ds),
                     'transforms are not equal')
        assert_equal_crs(get_crs(ds0), get_crs(ds),
                         'CRS are not equal')
        assert_equal(get_resolution(ds0), get_resolution(ds),
                     'resolutions are not equal')
        assert_equal(get_bounds(ds0), get_bounds(ds),
                     'bounds are not equal')
        assert_equal(get_extent(ds0), get_extent(ds),
                     'extents are not equal')


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_nrows(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    assert_equal(nrows(ds), kwargs['ny'])


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_ncols(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    assert_equal(ncols(ds), kwargs['nx'])


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_get_transform(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    bounds = get_bounds(ds)
    resx = (bounds.right - bounds.left) / (ds.dims['x'] - 1)
    resy = (bounds.bottom - bounds.top) / (ds.dims['y'] - 1)
    xoff = bounds.left
    yoff = bounds.top
    transform = Affine(resx, 0, xoff, 0, resy, yoff)
    assert_equal(
        get_transform(ds), transform
    )


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_get_crs(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    assert_equal_crs(get_crs(ds), kwargs['crs'])


@pytest.mark.parametrize('ds', slc_datasets)
def test_resolution_equal_transform_from_real_data(ds):
    res = get_resolution(ds)
    tf = get_transform(ds)
    assert_equal(res, (tf.a, abs(tf.e)))


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_get_resolution(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    res = get_resolution(ds)
    bounds = get_bounds(ds)
    resx = abs(bounds.right - bounds.left) / (ncols(ds) - 1)
    resy = abs(bounds.bottom - bounds.top) / (nrows(ds) - 1)
    assert_equal(res, (resx, resy))


def test_get_bounds():
    bounds = (-10.0, 50.0, 0.0, 60.0)
    ds = generate_test_dataset(extent=bounds)
    assert_equal(bounds, get_bounds(ds))


def test_get_common_extent():
    ...


def test_get_extent():
    extent = (-10.0, 50.0, 0.0, 60.0)
    ds = generate_test_dataset(extent=extent)
    assert_equal(extent, get_extent(ds))


# def test_warp_grid_shift():
#     ds = generate_test_dataset(ntime=1)

#     # [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat]
#     old_extent = [
#         ds['lon'].min(),
#         ds['lat'].min(),
#         ds['lon'].max(),
#         ds['lat'].max()
#     ]
#     new_extent = [
#         ds['lon'].min() + 2,
#         ds['lat'].min() + 2,
#         ds['lon'].max() + 2,
#         ds['lat'].max() + 2
#     ]
#     warped = warp.warp(ds, new_extent)
#     # Check that values outside old_extent are NaN


# def test_align(tmpdir):
#     path = tmpdir.mkdir('aligned')
#     # [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat]
#     extent1 = (-10, 50, 0, 60)
#     extent2 = (-8, 52, 2, 62)
#     ds1 = generate_test_dataset(extent=extent1)
#     ds2 = generate_test_dataset(extent=extent2)
#     warp.align([ds1, ds2], path)
#     # Check whether the aligned files have been created
#     assert_equal(os.listdir(path), ['data0_aligned.nc', 'data1_aligned.nc'])
#     # Open the aligned files
#     ds1_aligned = from_netcdf(str(path.join('data0_aligned.nc')))
#     ds2_aligned = from_netcdf(str(path.join('data1_aligned.nc')))
#     assert_equal(
#         warp._get_extent(ds1_aligned),
#         warp._get_extent(ds2_aligned)
#     )
#     xr_assert_equal(
#         ds1_aligned.lat, ds2_aligned.lat
#     )
#     xr_assert_equal(
#         ds1_aligned.lon, ds2_aligned.lon
#     )
