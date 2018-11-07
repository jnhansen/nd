import xarray as xr
from geo.utils import testing
from geo.change import OmnibusTest


def test_change():
    ds1 = testing.generate_test_dataset(
        mean=[1, 0, 0, 1], sigma=0.1, nlat=5, nlon=5
        ).isel(time=slice(None, 5))
    ds2 = testing.generate_test_dataset(
        mean=[10, 0, 0, 10], sigma=0.1, nlat=5, nlon=5
        ).isel(time=slice(5, None))
    ds = xr.concat([ds1, ds2], dim='time')
    changes = OmnibusTest(n=9).detect(ds, alpha=0.9)
    assert changes.isel(time=5).all()
    assert (changes.sum(dim='time') == 1).all()
