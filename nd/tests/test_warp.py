import pytest
from nd import warp
from nd.io import open_dataset, to_netcdf
from nd.testing import (generate_test_dataset, generate_test_dataarray,
                        assert_equal_crs)
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_raises,
                           assert_raises_regex, assert_array_almost_equal)
import xarray as xr
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical
import os
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.errors import CRSError
import rasterio.warp
from affine import Affine
from collections import OrderedDict
import shapely.geometry
import shapely.ops
import pyproj
from functools import partial


# Prepare test data
ds_params = [
    ('notime', {
        # Default
        'dims': {'y': 20, 'x': 10},
        'crs': CRS({'init': 'epsg:4326'})
    }),
    ('notime_mercator', {
        # Default
        'dims': {'y': 20, 'x': 10},
        'crs': CRS({'init': 'epsg:3395'})
    }),
    ('standard', {
        # Default
        'dims': {'y': 20, 'x': 10, 'time': 1},
        'crs': CRS({'init': 'epsg:4326'})
    }),
    ('standard_mercator', {
        # Test Mercator Projection
        'dims': {'y': 20, 'x': 10, 'time': 1},
        'crs': CRS({'init': 'epsg:3395'})
    }),
    ('ntime=5', {
        # Test temporal dimension
        'dims': {'y': 20, 'x': 10, 'time': 5},
        'crs': CRS({'init': 'epsg:4326'})
    }),
    ('variables', {
        # Test different variables
        'dims': {'y': 20, 'x': 10, 'time': 1},
        'crs': CRS({'init': 'epsg:4326'}),
        'var': ['v1', 'v2', 'v3']
    })
]

data_path = 'data/'
nc_path = os.path.join(data_path, 'slc.nc')
tif_path = os.path.join(data_path, 'slc.tif')
dim_path = os.path.join(data_path, 'slc.dim')
slc_files = [nc_path, tif_path, dim_path]
epsg4326 = CRS.from_epsg(4326)
sinusoidal = CRS.from_string('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 '
                             '+ellps=WGS84 +datum=WGS84 +units=m no_defs')


def create_snap_ds(*args, **kwargs):
    ds = generate_test_dataset(*args, **kwargs)
    crs = warp.get_crs(ds)
    t = warp.get_transform(ds)
    i2m_string = ','.join(map(str, [t.a, t.d, t.b, t.e, t.c, t.f]))
    del ds.attrs['crs']
    del ds.attrs['transform']
    ds['crs'] = ((), 1)
    attrs = {'crs': crs.wkt,
             'i2m': i2m_string}
    ds['crs'].attrs = attrs
    return ds


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_reprojection(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    crs = warp._parse_crs('epsg:4326')
    proj = warp.Reprojection(crs=crs)
    reprojected = proj.apply(ds)
    assert_equal_crs(crs, warp.get_crs(reprojected))


def test_reprojection_failure():
    ds = generate_test_dataset()
    transform = warp.get_transform(ds)
    extent = warp.get_extent(ds)
    with assert_raises_regex(
            ValueError, ".* must also specify the `width` and `height`.*"):
        _ = warp.Reprojection(crs=epsg4326, transform=transform)
    with assert_raises_regex(
            ValueError, "Need to provide either `width` and `height` .*"):
        _ = warp.Reprojection(crs=epsg4326, extent=extent)


def test_reprojection_with_src_crs():
    src_crs = epsg4326
    dst_crs = sinusoidal
    # Set up test dataset with and without CRS information
    ds = generate_test_dataset(crs=src_crs)
    assert_equal_crs(src_crs, ds.nd.crs)
    ds_nocrs = ds.copy()
    del ds_nocrs.attrs['crs']
    assert ds_nocrs.nd.crs is None

    with assert_raises_regex(
            CRSError,
            "Could not infer projection from input data. "
            "Please provide the parameter `src_crs`."):
        warp.Reprojection(dst_crs=dst_crs).apply(ds_nocrs)

    xr_assert_equal(
        warp.Reprojection(dst_crs=dst_crs).apply(ds),
        warp.Reprojection(src_crs=src_crs, dst_crs=dst_crs).apply(ds_nocrs)
    )


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_reprojection_with_target(generator):
    src_crs = epsg4326
    dst_crs = sinusoidal
    ds = generator(crs=src_crs)
    src_bounds = warp.get_bounds(ds)
    dst_bounds_latlon = BoundingBox(
        left=src_bounds.left - 1,
        bottom=src_bounds.bottom - 1,
        right=src_bounds.right + 1,
        top=src_bounds.top + 1,
    )
    dst_bounds = BoundingBox(*rasterio.warp.transform_bounds(
        src_crs, dst_crs, **dst_bounds_latlon._asdict()
    ))
    dst_width, dst_height = 35, 21
    resx = (dst_bounds.right - dst_bounds.left) / (dst_width - 1)
    resy = (dst_bounds.bottom - dst_bounds.top) / (dst_height - 1)
    res = (abs(resx), abs(resy))
    xoff = dst_bounds.left
    yoff = dst_bounds.top
    dst_transform = Affine(resx, 0, xoff, 0, resy, yoff)

    target = generator(
        dims={'x': dst_width, 'y': dst_height, 'time': 1},
        extent=dst_bounds, crs=dst_crs
    )

    projected = [
        warp.Reprojection(crs=dst_crs, transform=dst_transform,
                          width=dst_width, height=dst_height).apply(ds),
        warp.Reprojection(crs=dst_crs, extent=dst_bounds,
                          res=res).apply(ds),
        warp.Reprojection(crs=dst_crs, extent=dst_bounds,
                          width=dst_width, height=dst_height).apply(ds),
        warp.Reprojection(target=target).apply(ds),
    ]
    for i, proj in enumerate(projected[1:]):
        print(i)
        xr_assert_equal(proj, projected[0])
        assert_almost_equal(warp.get_resolution(proj), res)
        assert_almost_equal(warp.get_bounds(proj), dst_bounds)
        assert_almost_equal(warp.get_transform(proj), dst_transform)
        assert_equal_crs(warp.get_crs(proj), dst_crs)


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_resample_to_resolution_tuple(name, kwargs):
    res = (0.05, 0.01)
    ds = generate_test_dataset(**kwargs)
    resampled = warp.Resample(res=res).apply(ds)
    assert_almost_equal(res, warp.get_resolution(resampled))


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_resample_to_resolution_float(name, kwargs):
    res = 0.05
    ds = generate_test_dataset(**kwargs)
    resampled = warp.Resample(res=res).apply(ds)
    assert_almost_equal((res, res), warp.get_resolution(resampled))


@pytest.mark.parametrize('resample_kwargs', [{'width': 25}, {'height': 25}])
@pytest.mark.parametrize('name,kwargs', ds_params)
def test_resample_to_width_or_height(name, kwargs, resample_kwargs):
    ds = generate_test_dataset(**kwargs)
    resampled = warp.Resample(**resample_kwargs).apply(ds)
    if 'width' in resample_kwargs:
        assert_equal(resample_kwargs['width'], warp.ncols(resampled))
    elif 'height' in resample_kwargs:
        assert_equal(resample_kwargs['height'], warp.nrows(resampled))

    # Make sure aspect ratio is preserved
    assert_equal(
        int(warp.ncols(resampled) / warp.nrows(resampled)),
        int(warp.ncols(ds) / warp.nrows(ds))
    )


@pytest.mark.parametrize('crs', [
    CRS.from_string('epsg:4326')
])
def test_parse_crs(crs):
    assert_equal_crs(crs, warp._parse_crs(crs))
    assert_equal_crs(crs, warp._parse_crs(crs.to_string()))
    assert_equal_crs(crs, warp._parse_crs(crs.to_dict()))
    assert_equal_crs(crs, warp._parse_crs(crs.wkt))
    assert_equal_crs(crs, warp._parse_crs(crs.to_epsg()))


@pytest.mark.parametrize('invalidcrs', [
    'not_a_crs'
])
def test_parse_crs_fails(invalidcrs):
    with assert_raises(CRSError):
        warp._parse_crs(invalidcrs)


@pytest.mark.skip(reason="This currently fails due to SNAP saving "
                         "inconsistent datasets.")
def test_equal_datasets():
    ds0 = open_dataset(slc_files[0])
    for f in slc_files[1:]:
        ds = open_dataset(f)
        assert_equal(ds0['x'].values, ds['x'].values,
                     'x coordinates are not equal')
        assert_equal(ds0['y'].values, ds['y'].values,
                     'y coordinates are not equal')
        assert_equal(warp.get_transform(ds0), warp.get_transform(ds),
                     'transforms are not equal')
        assert_equal_crs(warp.get_crs(ds0), warp.get_crs(ds),
                         'CRS are not equal')
        assert_equal(warp.get_resolution(ds0), warp.get_resolution(ds),
                     'resolutions are not equal')
        assert_equal(warp.get_bounds(ds0), warp.get_bounds(ds),
                     'bounds are not equal')
        assert_equal(warp.get_extent(ds0), warp.get_extent(ds),
                     'extents are not equal')
        ds.close()
    ds0.close()


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_nrows(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    assert_equal(warp.nrows(ds), kwargs['dims']['y'])


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_ncols(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    assert_equal(warp.ncols(ds), kwargs['dims']['x'])


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_get_transform(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    x = ds.coords['x'].values
    y = ds.coords['y'].values
    resx = (x[-1] - x[0]) / (ds.dims['x'] - 1)
    resy = (y[-1] - y[0]) / (ds.dims['y'] - 1)
    xoff = x[0]
    yoff = y[0]
    transform = Affine(resx, 0, xoff, 0, resy, yoff)
    assert_equal(
        warp.get_transform(ds), transform
    )


# Test extraction of SNAP-style transform information.
@pytest.mark.parametrize('crs', [
    CRS.from_epsg(4326),
    CRS.from_epsg(3395),
])
def test_get_transform_from_variable(crs):
    ds = generate_test_dataset(crs=crs)
    snap_ds = create_snap_ds(crs=crs)
    assert_equal(
        warp._get_transform_from_metadata(ds),
        warp._get_transform_from_metadata(snap_ds)
    )


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_get_crs(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    assert_equal_crs(warp.get_crs(ds), kwargs['crs'])


def test_get_crs_none():
    ds = generate_test_dataset()
    ds.attrs = {}
    assert warp.get_crs(ds) is None


@pytest.mark.parametrize('fmt,result', [
    ('proj', '+init=epsg:4326'),
    ('dict', {'init': 'epsg:4326'}),
    ('wkt', epsg4326.wkt)
])
def test_get_crs_formats(fmt, result):
    ds = generate_test_dataset(crs=CRS.from_epsg(4326))
    assert_equal(warp.get_crs(ds, format=fmt), result)


# Test extraction of SNAP-style CRS information.
@pytest.mark.parametrize('crs', [
    CRS.from_epsg(4326),
    CRS.from_epsg(3395),
])
def test_get_crs_from_variable(crs):
    snap_ds = create_snap_ds(crs=crs)
    parsed_crs = warp.get_crs(snap_ds)
    assert_equal_crs(crs, parsed_crs)


@pytest.mark.parametrize('f', slc_files)
def test_resolution_equal_transform_from_real_data(f):
    ds = open_dataset(f)
    res = warp.get_resolution(ds)
    tf = warp.get_transform(ds)
    ds.close()
    assert_almost_equal(res, (tf.a, abs(tf.e)))


@pytest.mark.parametrize('name,kwargs', ds_params)
def test_get_resolution(name, kwargs):
    ds = generate_test_dataset(**kwargs)
    res = warp.get_resolution(ds)
    bounds = warp.get_bounds(ds)
    resx = abs(bounds.right - bounds.left) / (warp.ncols(ds) - 1)
    resy = abs(bounds.bottom - bounds.top) / (warp.nrows(ds) - 1)
    assert_almost_equal(res, (resx, resy))


def test_get_transform_from_metadata():
    ds = generate_test_dataset()
    assert_equal(
        warp._get_transform_from_metadata(ds),
        warp.get_transform(ds)
    )


def test_get_resolution_from_metadata():
    ds = generate_test_dataset()
    assert_equal(
        warp._get_resolution_from_metadata(ds),
        warp.get_resolution(ds)
    )
    del ds.attrs['transform']
    assert_equal(
        warp._get_resolution_from_metadata(ds),
        warp.get_resolution(ds)
    )
    del ds.attrs['res']
    assert_equal(
        warp._get_resolution_from_metadata(ds),
        None
    )


def test_get_bounds_from_metadata():
    ds = generate_test_dataset()
    assert_equal(
        warp._get_bounds_from_metadata(ds),
        warp.get_bounds(ds)
    )
    del ds.attrs['transform']
    assert_equal(
        warp._get_bounds_from_metadata(ds),
        warp.get_bounds(ds)
    )
    del ds.attrs['bounds']
    assert_equal(
        warp._get_bounds_from_metadata(ds),
        None
    )


def test_get_bounds_dataset():
    bounds = (-10.0, 50.0, 0.0, 60.0)
    ds = generate_test_dataset(extent=bounds)
    assert_equal(bounds, warp.get_bounds(ds))


def test_get_bounds_dataarray():
    bounds = (-10.0, 50.0, 0.0, 60.0)
    da = generate_test_dataarray(extent=bounds)
    assert_equal(bounds, warp.get_bounds(da))


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
@pytest.mark.parametrize('bounds', [
    (-10.0, 50.0, 0.0, 60.0),
    (3.0, 55.0, 5.0, 58.0)
])
@pytest.mark.parametrize('src_crs', [
    epsg4326, sinusoidal
])
@pytest.mark.parametrize('dst_crs', [
    CRS({'init': 'epsg:4326'}),
    CRS({'init': 'epsg:3395'})
])
def test_get_geometry(generator, bounds, src_crs, dst_crs):
    ds = generator(extent=bounds, crs=src_crs)
    box = shapely.geometry.box(*warp.get_bounds(ds))
    assert_equal(
        warp.get_geometry(ds, crs=src_crs), box)

    geom = warp.get_geometry(ds, crs=dst_crs)

    try:
        # pyproj >= 2.1.0
        project = pyproj.Transformer.from_crs(
            src_crs, dst_crs, always_xy=True).transform
        inverse_project = pyproj.Transformer.from_crs(
            dst_crs, src_crs, always_xy=True).transform

    except Exception:
        # pyproj < 2.1
        project = partial(
            pyproj.transform,
            warp._to_pyproj(src_crs),
            warp._to_pyproj(dst_crs))
        inverse_project = partial(
            pyproj.transform,
            warp._to_pyproj(dst_crs),
            warp._to_pyproj(src_crs))

    expected = shapely.ops.transform(project, box)
    assert_equal(geom, expected)

    expected_inverse = shapely.ops.transform(inverse_project, geom)
    assert expected_inverse.equals_exact(box, tolerance=1e-6)


def test_get_extent_dataset():
    extent = (-10.0, 50.0, 0.0, 60.0)
    ds = generate_test_dataset(extent=extent, crs=epsg4326)
    assert_equal(extent, warp.get_extent(ds))


def test_get_extent_dataarray():
    extent = (-10.0, 50.0, 0.0, 60.0)
    da = generate_test_dataarray(extent=extent, crs=epsg4326)
    assert_equal(extent, warp.get_extent(da))


def test_get_common_bounds():
    bounds = [
        (-10.0, 50.0, 0.0, 60.0),
        (-12.0, 40.0, -2.0, 52.0),
        (-13.0, 50.0, -3.0, 60.0),
        (-9.0, 51.0, 1.0, 61.0)
    ]
    datasets = [generate_test_dataset(extent=ext) for ext in bounds]
    assert_equal(
        warp.get_common_bounds(datasets),
        (-13.0, 40.0, 1.0, 61.0)
    )


def test_get_common_extent():
    bounds = [
        (-10.0, 50.0, 0.0, 60.0),
        (-12.0, 40.0, -2.0, 52.0),
        (-13.0, 50.0, -3.0, 60.0),
        (-9.0, 51.0, 1.0, 61.0)
    ]
    common_extent = (-13.0, 40.0, 1.0, 61.0)
    datasets = [generate_test_dataset(extent=ext) for ext in bounds]

    # Reproject such that the projected bounds change,
    # but the extent remains the same:
    proj = warp.Reprojection(crs=sinusoidal)
    datasets_sinu = [proj.apply(ds) for ds in datasets]

    common_bounds = warp.get_common_bounds(datasets_sinu)
    expected_result = BoundingBox(*rasterio.warp.transform_bounds(
        sinusoidal, epsg4326, **common_bounds._asdict()
    ))

    assert_raises(AssertionError, assert_equal,
                  common_bounds, common_extent)
    assert_almost_equal(warp.get_common_extent(datasets_sinu),
                        expected_result)


@pytest.mark.parametrize('mode,fn', [
    ('min', np.min),
    ('max', np.max),
    ('mean', np.mean)
])
def test_get_common_resolution(mode, fn):
    bounds = [
        (-10.0, 50.0, 0.0, 60.0),
        (-12.0, 40.0, -2.0, 52.0),
        (-13.0, 50.0, -3.0, 60.0),
        (-9.0, 51.0, 1.0, 61.0)
    ]
    datasets = [generate_test_dataset(extent=ext) for ext in bounds]
    res = np.array([warp.get_resolution(ds) for ds in datasets])
    common_res = tuple(fn(res, axis=0))
    assert_equal(warp.get_common_resolution(datasets, mode=mode),
                 common_res)


def test_get_common_resolution_invalid_mode():
    datasets = [generate_test_dataset() for i in range(3)]
    with assert_raises_regex(ValueError,
                             "Unsupported mode: 'invalid'"):
        warp.get_common_resolution(datasets, mode='invalid')


def test_get_common_resolution_different_projections():
    crs = [epsg4326, sinusoidal]
    datasets = [generate_test_dataset(crs=c) for c in crs]
    with assert_raises_regex(ValueError,
                             "All datasets must have the same projection."):
        warp.get_common_resolution(datasets)


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_get_dims(generator):
    dims = {'x': 5, 'y': 10, 'time': 15}
    ds = generator(dims=dims)
    assert_equal(warp.get_dim_sizes(ds), dims)


@pytest.mark.parametrize('extra,dims', [
    (('y',), OrderedDict([('x', 20)])),
    (('x',), OrderedDict([('y', 30)])),
    ((), OrderedDict([('y', 30), ('x', 20)])),
    ((), OrderedDict([('y', 30), ('x', 20), ('time', 5)]))
])
def test_expand_var_to_xy(extra, dims):
    ref = generate_test_dataarray(
        dims=OrderedDict([('y', 30), ('x', 20), ('time', 5)])
    )
    da = generate_test_dataarray(dims=dims)
    expanded = warp._expand_var_to_xy(da, ref.coords)

    # Check that the variance along added dimensions is zero
    assert np.all(
        expanded.var(extra) < 1e-16
    )

    # Check that the new DataArray contains x and y coordinates
    xr_assert_equal(
        expanded.coords['x'], ref.coords['x'])
    xr_assert_equal(
        expanded.coords['y'], ref.coords['y'])

    # Check that the DataArray is unchanged if x and y were already
    # dimensions
    if 'x' in dims and 'y' in dims:
        xr_assert_equal(da, expanded)


def test_collapse_coords():
    ds = generate_test_dataset()
    for c in ds.coords:
        expanded = xr.concat([ds.coords[c]] * 10, dim='new')
        xr_assert_equal(
            ds.coords[c], warp._collapse_coords(expanded)
        )


def test_expand_collapse_coords():
    ds = generate_test_dataset()
    for c in ['x', 'y']:
        expanded = warp._expand_var_to_xy(ds.coords[c], ds.coords)
        xr_assert_equal(
            ds.coords[c], warp._collapse_coords(expanded)
        )


def test_reproject_no_hidden_effects():
    src_crs = epsg4326
    dst_crs = sinusoidal
    ds = generate_test_dataset(crs=src_crs)
    ds_copy = ds.copy(deep=True)
    _ = warp._reproject(ds_copy, dst_crs=dst_crs)
    xr_assert_identical(ds, ds_copy)


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_reproject(generator):
    src_crs = epsg4326
    dst_crs = sinusoidal
    ds = generator(crs=src_crs)
    src_bounds = warp.get_bounds(ds)
    dst_bounds_latlon = BoundingBox(
        left=src_bounds.left - 1,
        bottom=src_bounds.bottom - 1,
        right=src_bounds.right + 1,
        top=src_bounds.top + 1,
    )
    dst_bounds = BoundingBox(*rasterio.warp.transform_bounds(
        src_crs, dst_crs, **dst_bounds_latlon._asdict()
    ))
    dst_width, dst_height = 35, 21
    resx = (dst_bounds.right - dst_bounds.left) / (dst_width - 1)
    resy = (dst_bounds.bottom - dst_bounds.top) / (dst_height - 1)
    res = (abs(resx), abs(resy))
    xoff = dst_bounds.left
    yoff = dst_bounds.top
    dst_transform = Affine(resx, 0, xoff, 0, resy, yoff)

    projected = [
        warp._reproject(ds, dst_crs=dst_crs, dst_transform=dst_transform,
                        width=dst_width, height=dst_height),
        warp._reproject(ds, dst_crs=dst_crs, dst_transform=dst_transform,
                        extent=dst_bounds),
        warp._reproject(ds, dst_crs=dst_crs, extent=dst_bounds,
                        res=res),
        warp._reproject(ds, dst_crs=dst_crs, extent=dst_bounds,
                        width=dst_width, height=dst_height),
    ]
    for proj in projected[1:]:
        xr_assert_equal(proj, projected[0])
        assert_almost_equal(warp.get_resolution(proj), res)
        assert_almost_equal(warp.get_bounds(proj), dst_bounds)
        assert_almost_equal(warp.get_transform(proj), dst_transform)
        assert_equal_crs(warp.get_crs(proj), dst_crs)


def test_reproject_insufficient_information():
    src_crs = epsg4326
    ds = generate_test_dataset(crs=src_crs)
    dst_transform = Affine(0.1, 0, -10, 0, 0.1, 49)

    insufficient_kwargs = [
        dict(dst_transform=dst_transform),
    ]
    for kwargs in insufficient_kwargs:
        with assert_raises(ValueError):
            _ = warp._reproject(ds, **kwargs)


def test_reproject_one_dimensional_vars():
    ds = generate_test_dataset(crs=epsg4326)
    ds['xvar'] = (('x',), ds.x.values * 10)
    ds['yvar'] = (('y',), ds.y.values * 10)
    ds['timevar'] = (('time',), np.random.rand(ds.dims['time']))

    warped = warp.Reprojection(
        crs=sinusoidal, resampling=rasterio.warp.Resampling.bilinear
    ).apply(ds)

    xr_assert_equal(
        warped['timevar'], ds['timevar'])

    # Check that xvar and yvar are still proportional to longitude
    # and latitude
    for v, coord in [
        ('xvar', 'lon'), ('yvar', 'lat')
    ]:
        ratios = warped[v].values / warped[coord].values
        print(v, coord)
        print(np.nanmin(ratios), np.nanmax(ratios), np.nanstd(ratios))
        # There is quite a bit of error, for now accept relatively
        # high error.
        assert np.nanstd(ratios) < 1.5


@pytest.mark.skip(
    reason="currently failing because of an inconsistency in "
    "rasterio: https://github.com/mapbox/rasterio/issues/1934."
)
def test_reproject_one_dimensional_coords():
    ds = generate_test_dataset(crs=epsg4326)
    # Add coordinates that are not dimensions
    ds.coords['longitude'] = (('x',), ds.x.values)
    ds.coords['latitude'] = (('y',), ds.y.values)

    warped = warp.Reprojection(
        crs=warp.get_crs(ds), width=15, height=15,
        resampling=rasterio.warp.Resampling.bilinear
    ).apply(ds)

    # Check that xvar and yvar are still proportional to longitude
    # and latitude
    for coord, dim in [
        ('longitude', 'x'), ('latitude', 'y')
    ]:
        assert_array_almost_equal(
            warped.coords[coord], warped.coords[dim])


def test_reprojection_nan_values():
    src_crs = epsg4326
    dst_crs = sinusoidal
    ds = generate_test_dataset(crs=src_crs)
    bounds = warp.get_bounds(ds)
    proj = warp.Reprojection(crs=dst_crs)
    warped = proj.apply(ds)
    xgrid, ygrid = np.meshgrid(warped.x, warped.y)
    lon, lat = rasterio.warp.transform(dst_crs, src_crs, xgrid.flatten(),
                                       ygrid.flatten())
    lon = np.array(lon).reshape(xgrid.shape)
    lat = np.array(lat).reshape(ygrid.shape)

    inside_bounds = np.logical_and(
        np.logical_and(lon >= bounds.left, lon <= bounds.right),
        np.logical_and(lat >= bounds.bottom, lat <= bounds.top)
    )
    for v in warped.data_vars:
        if not set(warped[v].dims).issuperset({'y', 'x'}):
            continue
        dim_order = tuple(set(warped[v].dims) - {'y', 'x'}) + ('y', 'x')
        values = warped[v].transpose(*dim_order, transpose_coords=True).values
        # Check that pixels strictly inside the original bounds are not NaN
        assert np.isnan(values[..., inside_bounds]).sum() == 0
        # Pixel outside of the original bounds should be mostly NaN,
        # although some pixels near the edges may have values.
        outside_values = values[..., ~inside_bounds]
        assert np.isnan(outside_values).sum() / outside_values.size > 0.5


def test_reproject_coordinates():
    ds = generate_test_dataset(crs=epsg4326)
    dims = warp.get_dim_sizes(ds)
    ds.coords['lat'] = ds['y']
    ds.coords['lon'] = ds['x']
    ds.coords['altitude'] = (('y', 'x'),
                             np.zeros((dims['y'], dims['x'])))
    proj = warp.Reprojection(crs=sinusoidal)
    warped = proj.apply(ds)
    for c in ds.coords:
        if c in ['lat', 'lon']:
            continue
        assert c in warped.coords
        assert_equal(sorted(ds[c].dims), sorted(warped[c].dims))


@pytest.mark.parametrize('extent', [
    None, (-10.0, 50.0, 0.0, 60.0)
])
@pytest.mark.parametrize('from_files', [True, False])
def test_alignment(tmpdir, extent, from_files):
    datapath = tmpdir.mkdir('data')
    path = tmpdir.mkdir('aligned')
    bounds = [
        (-10.0, 50.0, 0.0, 60.0),
        (-12.0, 40.0, -2.0, 52.0),
        (-13.0, 50.0, -3.0, 60.0),
        (-9.0, 51.0, 1.0, 61.0)
    ]
    datasets = [generate_test_dataset(extent=ext) for ext in bounds]
    if extent is None:
        common_bounds = warp.get_common_bounds(datasets)
    else:
        common_bounds = extent
    files = [str(datapath.join('data_%d.nc' % i))
             for i in range(len(datasets))]
    if from_files:
        for ds, f in zip(datasets, files):
            to_netcdf(ds, f)
        datasets = files
    warp.Alignment(extent=extent).apply(datasets, path=str(path))
    aligned = [open_dataset(str(f)) for f in path.listdir()]
    for ds in aligned:
        assert_equal(warp.get_bounds(ds), common_bounds)
        assert_equal(
            warp.get_transform(ds),
            warp.get_transform(aligned[0])
        )
        xr_assert_equal(ds['x'], aligned[0]['x'])
        xr_assert_equal(ds['y'], aligned[0]['y'])


@pytest.mark.parametrize('dims', [
    {'y': 20, 'x': 20, 'time': 10, 'band': 5},
    {'x': 20, 'y': 20, 'time': 10, 'band': 5},
    {'time': 10, 'band': 5, 'x': 20, 'y': 20},
    {'time': 10, 'x': 20, 'band': 5, 'y': 20},
    {'y': 20, 'x': 20, 'time': 10, 'band': 5, 'extra': 2}
])
def test_reproject_with_extra_dims(dims):
    crs1 = warp._parse_crs('epsg:4326')
    crs2 = warp._parse_crs('epsg:3395')
    ds = generate_test_dataset(
        dims=dims, crs=crs1
    )

    proj = warp.Reprojection(crs=crs2)
    reprojected = proj.apply(ds)

    # Check that a reprojected slice of the dataset is the same as
    # the slice of the reprojection of the entire dataset.
    slices = [
        {'band': 3},
        {'time': slice(1, 3)}
    ]
    for s in slices:
        xr_assert_equal(
            proj.apply(ds.isel(**s)),
            reprojected.isel(**s)
        )


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
