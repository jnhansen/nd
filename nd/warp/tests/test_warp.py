import pytest
from nd import warp
from nd.io import from_netcdf
from nd.testing import generate_test_dataset
import numpy as np
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
import os
from rasterio.crs import CRS


ds = generate_test_dataset(ntime=1)


def test_Reprojection():
    proj = warp.Reprojection(crs='+init=epsg:4326')


def test_Alignment():
    ...


@pytest.mark.parametrize('crs', [
    CRS.from_string('+init=epsg:4326')
])
def test_parse_crs(crs):
    assert crs == warp._parse_crs(crs)
    assert crs == warp._parse_crs(crs.to_string())
    assert crs == warp._parse_crs(crs.to_dict())
    assert crs == warp._parse_crs(crs.wkt)


def test_get_crs():
    ...


def test_get_transform():
    ...


def test_get_resolution():
    ...


def test_get_bounds():
    extent = (-10.0, 50.0, 0.0, 60.0)
    ds = generate_test_dataset(extent=extent)
    print(extent, warp.get_bounds(ds))
    assert_equal(extent, warp.get_bounds(ds))


def test_get_common_extent():
    ...


def test_get_extent():
    extent = (-10.0, 50.0, 0.0, 60.0)
    ds = generate_test_dataset(extent=extent)
    assert_equal(extent, warp.get_extent(ds))


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
