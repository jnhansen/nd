import pytest
import os
import xarray as xr
from nd.io import (open_dataset, open_netcdf, open_beam_dimap, open_rasterio,
                   to_netcdf, assemble_complex)
from nd.testing import generate_test_dataset
from xarray.testing import assert_equal as xr_assert_equal


data_path = 'data/'
nc_path = os.path.join(data_path, 'slc.nc')
tif_path = os.path.join(data_path, 'slc.tif')
dim_path = os.path.join(data_path, 'slc.dim')


@pytest.mark.parametrize('f', [nc_path, tif_path, dim_path])
def test_open_dataset(f):
    ds = open_dataset(f)
    assert isinstance(ds, (xr.Dataset, xr.DataArray))
    ds.close()


def test_open_netcdf():
    ds = open_netcdf(nc_path)
    assert isinstance(ds, xr.Dataset)
    ds.close()


def test_open_beam_dimap():
    ds = open_beam_dimap(dim_path)
    assert isinstance(ds, xr.Dataset)
    ds.close()


def test_open_rasterio():
    ds = open_rasterio(tif_path)
    assert isinstance(ds, xr.DataArray)


@pytest.mark.skip
def test_equivalent_formats():
    files = [nc_path, tif_path, dim_path]
    datasets = [open_dataset(f) for f in files]


@pytest.mark.parametrize('cmplx', [True, False])
def test_write_read_netcdf(tmpdir, cmplx):
    ds = generate_test_dataset()
    if cmplx:
        ds = assemble_complex(ds)
    path = str(tmpdir.join('test_dataset.nc'))
    to_netcdf(ds, path)
    ds_read = open_dataset(path, as_complex=cmplx)
    xr_assert_equal(ds, ds_read)
