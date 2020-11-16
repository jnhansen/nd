import xarray as xr
from nd import testing
from nd.change import OmnibusTest


@testing.requires('gsl')
def test_change():
    ds1 = testing.generate_test_dataset(
        dims={'y': 5, 'x': 5, 'time': 10},
        mean=[1, 0, 0, 1], sigma=0.1
        ).isel(time=slice(None, 5))
    ds2 = testing.generate_test_dataset(
        dims={'y': 5, 'x': 5, 'time': 10},
        mean=[10, 0, 0, 10], sigma=0.1
        ).isel(time=slice(5, None))
    ds = xr.concat([ds1, ds2], dim='time')
    changes = OmnibusTest(n=9, alpha=0.9).apply(ds)
    assert changes.isel(time=5).all()
    assert (changes.sum(dim='time') == 1).all()
