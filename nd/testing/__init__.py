import numpy as np
import pandas as pd
import xarray as xr
import json
import pkgutil
import inspect
import nd
import rasterio.transform
import rasterio.warp
from numpy.testing import assert_equal, assert_almost_equal
from nd.algorithm import Algorithm
from nd.warp.warp_ import _parse_crs


def generate_test_dataset(ny=20, nx=20, ntime=10,
                          var=['C11', 'C12__im', 'C12__re', 'C22'],
                          mean=0, sigma=1,
                          extent=(-10.0, 50.0, 0.0, 60.0),
                          random_seed=42,
                          crs='+init=epsg:4326'):

    np.random.seed(random_seed)
    coords = {}
    dims = {}
    if ny:
        coords['y'] = np.linspace(extent[1], extent[3], ny)
        dims['y'] = ny
    if nx:
        coords['x'] = np.linspace(extent[0], extent[2], nx)
        dims['x'] = nx
    if ntime:
        coords['time'] = pd.date_range('2017-01-01', '2018-01-01',
                                       periods=ntime)
        dims['time'] = ntime

    meta = {'attr1': 1, 'attr2': 2, 'attr3': 3}
    ds = xr.Dataset(coords=coords, attrs=meta)
    transform = rasterio.transform.from_bounds(
        *extent, width=nx-1, height=ny-1)
    ds.attrs['crs'] = _parse_crs(crs).to_string()
    ds.attrs['transform'] = transform[:6]
    if isinstance(mean, (int, float)):
        mean = [mean] * len(var)
    for v, m in zip(var, mean):
        ds[v] = (tuple(dims.keys()),
                 np.random.normal(m, sigma, tuple(dims.values())))
    return ds


def generate_test_dataarray(ny=20, nx=20, ntime=10, name='variable',
                            mean=0, sigma=1,
                            extent=(-10.0, 50.0, 0.0, 60.0),
                            random_seed=42,
                            crs='+init=epsg:4326'):
    np.random.seed(random_seed)
    coords = {}
    dims = {}
    if ny:
        coords['y'] = np.linspace(extent[1], extent[3], ny)
        dims['y'] = ny
    if nx:
        coords['x'] = np.linspace(extent[0], extent[2], nx)
        dims['x'] = nx
    if ntime:
        coords['time'] = pd.date_range('2017-01-01', '2018-01-01',
                                       periods=ntime)
        dims['time'] = ntime

    meta = {'attr1': 1, 'attr2': 2, 'attr3': 3}
    transform = rasterio.transform.from_bounds(
        *extent, width=nx-1, height=ny-1)
    meta['crs'] = _parse_crs(crs).to_string()
    meta['transform'] = transform[:6]
    data = np.random.normal(mean, sigma, tuple(dims.values()))
    da = xr.DataArray(data, coords=coords, dims=list(dims.keys()),
                      name=name, attrs=meta)
    return da


def equal_list_of_dicts(obj1, obj2, exclude=[]):
    """Check whether two lists of dictionaries are equal, independent of the
    order within the list.

    Parameters
    ----------
    obj1 : list of dict
        First object to compare
    obj2 : list of dict
        Second object to compare
    exclude : list of str, optional
        A list of keys that are to be excluded when deciding whether the lists
        of dictionaries are equal (default: []).

    Returns
    -------
    bool
        True if the two lists contain the same dictionaries, False otherwise.
    """

    for key in exclude:
        for i in obj1:
            del i[key]
        for i in obj2:
            del i[key]
    serial1 = [json.dumps(_, sort_keys=True) for _ in obj1]
    serial2 = [json.dumps(_, sort_keys=True) for _ in obj2]

    return set(serial1) == set(serial2)


def assert_all_true(ds):
    assert ds.to_array().values.all()


def assert_equal_data(ds1, ds2):
    """Assert that the data in two datasets is the same, independent
    of the order of the dimensions.
    """
    assert_all_true(ds1 - ds2 == 0)


def assert_equal_crs(crs1, crs2, *args, **kwargs):
    xs = np.arange(10, dtype=np.float64)
    ys = np.arange(10, dtype=np.float64)
    newx, newy = rasterio.warp.transform(crs1, crs2, xs, ys)
    assert_almost_equal(xs, np.array(newx), 6, *args, **kwargs)
    assert_almost_equal(ys, np.array(newy), 6, *args, **kwargs)

# def generate_test_latlon_grid(shape):
#     y, x = np.meshgrid(np.arange(shape[1]),
#                        np.arange(shape[0]),
#                        copy=False)
#     xy = np.stack((x, y), axis=-1).reshape((-1, 2))
#     p = PolynomialFeatures(degree=1)
#     xy_poly = p.fit_transform(xy)
#     n_coefs = xy_poly.shape[1]
#     coefs_lat = np.random.normal(size=n_coefs)
#     coefs_lon = np.random.normal(size=n_coefs)
#     lat = (xy_poly * coefs_lat).sum(axis=1).reshape(shape)
#     lon = (xy_poly * coefs_lon).sum(axis=1).reshape(shape)
#     return lat, lon


def all_algorithms(parent=nd):
    """
    Return a list of all algorithms.
    """
    all_classes = []
    path = parent.__path__
    prefix = parent.__name__ + '.'

    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix=prefix, onerror=lambda x: None):
        if (".tests." in modname):
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    algorithms = [c for c in all_classes
                  if (issubclass(c[1], Algorithm) and
                      c[0] != 'Algorithm')]

    return algorithms
