import pytest
import os
import xarray as xr
from nd.io import open_dataset, open_netcdf, open_beam_dimap, open_rasterio
from nd.testing import generate_test_dataset


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
