import xarray as xr
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical
from nd.testing import generate_test_dataset, generate_test_dataarray
from nd import warp


def test_dataset_accessor_reproject():
    ds = generate_test_dataset()
    kwargs = dict(crs='epsg:27700')
    xr_assert_equal(
        warp.reproject(ds, **kwargs),
        ds.nd.reproject(**kwargs)
    )


def test_dataarray_accessor_reproject():
    da = generate_test_dataarray()
    kwargs = dict(crs='epsg:27700')
    xr_assert_equal(
        warp.reproject(da, **kwargs),
        da.nd.reproject(**kwargs)
    )
