import numpy as np
import pandas as pd
import xarray as xr
import json


def generate_test_dataset(nlat=20, nlon=20, ntime=10,
                          var=['C11', 'C12__im', 'C12__re', 'C22'],
                          mean=0, sigma=1,
                          extent=(-10.0, 50.0, 0.0, 60.0),
                          random_seed=42):

    np.random.seed(random_seed)
    lats = np.linspace(extent[1], extent[3], nlat)
    lons = np.linspace(extent[0], extent[2], nlon)
    meta = {'attr1': 1, 'attr2': 2, 'attr3': 3}
    times = pd.date_range('2017-01-01', '2018-01-01', periods=ntime)
    ds = xr.Dataset(coords={'lat': lats, 'lon': lons, 'time': times},
                    attrs=meta)
    if isinstance(mean, (int, float)):
        mean = [mean] * len(var)
    for v, m in zip(var, mean):
        ds[v] = (('lat', 'lon', 'time'),
                 np.random.normal(m, sigma, (nlat, nlon, ntime)))
    return ds


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
