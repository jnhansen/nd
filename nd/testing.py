import pytest
import numpy as np
import pandas as pd
import xarray as xr
import json
import pkgutil
import inspect
import nd
import rasterio.transform
import rasterio.warp
import shapely
import geopandas as gpd
import pytest
from numpy.testing import assert_almost_equal
from nd.algorithm import Algorithm
from nd.warp import _parse_crs
from nd.utils import check_requirements
import hashlib
import os
from collections import OrderedDict


def requires(dependency):
    """Decorator to be used for skipping tests if the dependency
    requirements are not met.
    """
    available = check_requirements(dependency)
    return pytest.mark.skipif(
        not available,
        reason="This test requires the following "
               "dependencies: {}".format(dependency))


def generate_test_dataset(
        dims=OrderedDict([('y', 20), ('x', 20), ('time', 10)]),
        var=['C11', 'C12__im', 'C12__re', 'C22'],
        mean=0, sigma=1,
        extent=(-10.0, 50.0, 0.0, 60.0),
        random_seed=42,
        crs='epsg:4326'):

    np.random.seed(random_seed)

    coords = OrderedDict()
    for name, size in dims.items():
        if name == 'y':
            coords[name] = np.linspace(extent[3], extent[1], size)
        elif name == 'x':
            coords[name] = np.linspace(extent[0], extent[2], size)
        elif name == 'time':
            coords[name] = pd.date_range('2017-01-01', '2018-01-01',
                                         periods=size)
        else:
            coords[name] = np.arange(size)

    meta = {'attr1': 1, 'attr2': 2, 'attr3': 3}
    ds = xr.Dataset(coords=coords, attrs=meta)
    if 'x' in dims and 'y' in dims:
        transform = rasterio.transform.from_bounds(
            *extent, width=dims['x']-1, height=dims['y']-1)
        ds.attrs['crs'] = _parse_crs(crs).to_string()
        ds.attrs['transform'] = transform[:6]
        ds.attrs['res'] = (abs(transform.a), abs(transform.e))
        ds.attrs['bounds'] = extent
    if isinstance(mean, (int, float)):
        mean = [mean] * len(var)
    for v, m in zip(var, mean):
        ds[v] = (tuple(dims.keys()),
                 np.random.normal(m, sigma, tuple(dims.values())))
    return ds


def generate_test_dataarray(
        dims=OrderedDict([('y', 20), ('x', 20), ('time', 10)]),
        name='variable',
        mean=0, sigma=1,
        extent=(-10.0, 50.0, 0.0, 60.0),
        random_seed=42,
        crs='epsg:4326'):
    np.random.seed(random_seed)

    coords = OrderedDict()
    if 'y' in dims:
        coords['y'] = np.linspace(extent[3], extent[1], dims['y'])
    if 'x' in dims:
        coords['x'] = np.linspace(extent[0], extent[2], dims['x'])
    if 'time' in dims:
        coords['time'] = pd.date_range('2017-01-01', '2018-01-01',
                                       periods=dims['time'])

    meta = {'attr1': 1, 'attr2': 2, 'attr3': 3}
    if 'x' in dims and 'y' in dims:
        transform = rasterio.transform.from_bounds(
            *extent, width=dims['x']-1, height=dims['y']-1)
        meta['crs'] = _parse_crs(crs).to_string()
        meta['transform'] = transform[:6]
    data = np.random.normal(mean, sigma, tuple(dims.values()))
    da = xr.DataArray(data, coords=coords, dims=list(dims.keys()),
                      name=name, attrs=meta)
    return da


def create_mock_classes(dims):
    shape = (dims['y'], dims['x'])
    ds = generate_test_dataset(
        dims=dims,
        mean=[1, 0, 0, 1], sigma=0.1)
    ds2 = generate_test_dataset(
        dims=dims,
        mean=[10, 0, 0, 10], sigma=0.1)
    mask = np.zeros(shape, dtype=bool)
    mask = xr.DataArray(np.zeros(shape, dtype=bool),
                        dims=('y', 'x'),
                        coords=dict(y=ds.y, x=ds.x))

    # Make half of the data belong to each class
    mask[:, :dims['x']//2] = True
    ds = ds.where(mask, ds2)
    labels_true = (mask * 2).where(mask, 1)
    return ds, labels_true


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


def assert_equal_dict(dict1, dict2):
    """Check whether two lists of dictionaries are equal, independent of the
    order within the list.

    Parameters
    ----------
    dict1, dict2 : dict
        Dictioniaries to compare
    """

    j1 = json.dumps(dict1, sort_keys=True)
    j2 = json.dumps(dict2, sort_keys=True)
    assert j1 == j2


def assert_all_true(ds):
    assert ds.to_array().values.all()


def assert_equal_data(ds1, ds2):
    """Assert that the data in two datasets is the same, independent
    of the order of the dimensions.
    """
    assert_all_true(ds1 - ds2 == 0)


def assert_equal_crs(crs1, crs2, *args, **kwargs):
    # Perform some easy checks first.
    # If they fail check if a transform between both CRS
    # is approximately the identity.
    if crs1 is None and crs2 is None:
        return
    if crs1.to_wkt() == crs2.to_wkt():
        return
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


def _get_classes_from_module(modname):
    module = __import__(modname, fromlist="dummy")
    classes = inspect.getmembers(module, inspect.isclass)
    return classes


def all_algorithms(parent=nd):
    """
    Return a list of all algorithms.
    """
    if hasattr(parent, '__path__'):
        # The given module is a folder
        all_classes = []
        path = parent.__path__
        prefix = parent.__name__ + '.'

        for importer, modname, ispkg in pkgutil.walk_packages(
                path=path, prefix=prefix, onerror=lambda x: None):
            if (".tests." in modname):
                continue
            all_classes.extend(_get_classes_from_module(modname))

    else:
        # The given module is a file
        all_classes = _get_classes_from_module(parent.__name__)

    all_classes = set(all_classes)

    algorithms = [c for c in all_classes
                  if (issubclass(c[1], Algorithm) and
                      c[0] != 'Algorithm')]

    return algorithms


def _md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def assert_equal_files(*files):
    for f in files:
        assert os.path.isfile(f)
    checksums = [_md5(f) for f in files]
    for c in checksums[1:]:
        assert c == checksums[0]


# ----------------------
# Vector testing methods
# ----------------------

def random_polygon(x, y, n_vertices, radius=1, irregularity=0.2, sigma=0.3):
    """
    Generate a random polygon around given center.
    Inspired by https://stackoverflow.com/a/25276331/6156397

    Parameters
    ----------
    x, y : float
        Coordinates of polygon center
    n_vertices : int
        Number of polygon vertices
    radius : float, optional
        Average radius of polygon (default: 1)
    irregularity : float, optional
        Irregularity of spacing between the vertices (0..1)
        (default: 0.2)
    sigma : float, optional
        Standard deviation of radius, as fraction of radius (0..1)
        (default: 0.3)

    """

    irregularity = np.clip(irregularity, 0, 1) * 2*np.pi / n_vertices
    sigma = np.clip(sigma, 0, 1) * radius

    # Generate random angle increments for each vertex
    lower = (2*np.pi / n_vertices) - irregularity
    upper = (2*np.pi / n_vertices) + irregularity
    angle_steps = np.random.rand(n_vertices) * (upper-lower) + lower

    # Normalize the angle steps to fit exactly in a circle
    angle_steps = angle_steps * (2*np.pi) / angle_steps.sum()

    # Generate the points accordingly and create polygon
    angles = np.cumsum(angle_steps) + np.random.rand() * 2*np.pi
    radii = np.clip(sigma * np.random.randn(n_vertices) + radius, 0, 2*radius)
    vertices_x = radii * np.cos(angles) + x
    vertices_y = radii * np.sin(angles) + y
    polygon = shapely.geometry.Polygon(
        zip(vertices_x, vertices_y)
    )
    return polygon


def generate_test_polygons(n_polygon=20,
                           extent=(-10.0, 50.0, 0.0, 60.0),
                           radius=1,
                           crs='epsg:4326',
                           random_seed=None,
                           overlap=False):
    np.random.seed(random_seed)
    poly = []
    union = None
    while len(poly) < n_polygon:
        # Generate single polygon
        x = np.random.rand() * (extent[2] - extent[0]) + extent[0]
        y = np.random.rand() * (extent[3] - extent[1]) + extent[1]
        n = np.random.randint(3, 6)
        r = np.random.rand() * 1 + radius
        polygon = random_polygon(x, y, n, radius=r)

        if union is None:
            union = polygon
        else:
            # All sorts of topological errors can happen here,
            # just skip and move on
            try:
                if not overlap:
                    polygon = polygon.difference(union)

                    # If the result is a MultiPolygon,
                    # select only the largest part
                    if isinstance(polygon, shapely.geometry.MultiPolygon):
                        polygon = max(list(polygon), key=lambda p: p.area)

                    if not isinstance(polygon, shapely.geometry.Polygon) \
                            or not polygon.is_valid:
                        continue
                union = union.union(polygon)

            except shapely.errors.TopologicalError:
                continue

        if not polygon.is_empty:
            poly.append(polygon)

    return poly


def generate_test_geodataframe(n_polygon=20,
                               extent=(-10.0, 50.0, 0.0, 60.0),
                               radius=1,
                               crs='epsg:4326',
                               random_seed=None,
                               overlap=False):
    category_list = ['apple', 'pear', 'orange', 'banana']
    date_list = pd.date_range(start='01-2018', end='01-2019', freq='M').date
    poly = generate_test_polygons(n_polygon, radius=radius, overlap=overlap)
    df = gpd.GeoDataFrame({
        'category': np.random.choice(category_list, n_polygon),
        'float': np.random.rand(n_polygon),
        'integer': np.random.randint(0, 100, n_polygon),
        'date': np.random.choice(date_list, n_polygon),
        'geometry': poly
    })
    return df
