import numpy as np
import pandas as pd
import xarray as xr


def generate_test_dataset(nlat=20, nlon=20, ntime=10,
                          var=['C11', 'C12__im', 'C12__re', 'C22'],
                          mean=0, sigma=1):

    np.random.seed(42)
    lats = np.linspace(50.0, 60.0, nlat)
    lons = np.linspace(-10.0, 0.0, nlon)
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
