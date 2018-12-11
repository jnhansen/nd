import pytest
import numpy as np
import xarray as xr
from nd.testing import generate_test_dataset, assert_equal_data
from nd.utils import dict_product
from nd.io import open_dataset
from nd import tiling
from xarray.testing import assert_equal as xr_assert_equal
from numpy.testing import assert_equal


# Synthesize test data
ny = 20
nx = 20
ntime = 10
ds = generate_test_dataset(ny=ny, nx=nx, ntime=ntime)

# Generate tiles along y, x, and time dimension
slices = dict(y=[slice(None, 10), slice(10, None)],
              x=[slice(None, 10), slice(10, None)],
              time=[slice(None, 5), slice(5, None)])
parts = [ds.isel(**sl) for sl in dict_product(slices)]

# Generate tiles along y, x, and time dimension with
# buffer along y and x.
buffered_slices = dict(y=[slice(None, 12), slice(8, None)],
                       x=[slice(None, 11), slice(9, None)],
                       time=[slice(None, 5), slice(5, None)])
buffered_parts = [ds.isel(**sl) for sl in dict_product(buffered_slices)]


def test_auto_merge():
    xr_assert_equal(ds, tiling.auto_merge(parts))


def test_auto_merge_with_buffer():
    xr_assert_equal(ds, tiling.auto_merge(buffered_parts))


@pytest.mark.parametrize('buffer', [
    0, 2, {'x': 3}
])
@pytest.mark.parametrize('chunks', [
    {'time': 2},
    {'x': 4},
    {'y': 10, 'x': 10},
    {'y': 100, 'x': 100},
    {'y': 100, 'x': 8, 'time': 3},
])
def test_tile(tmpdir, chunks, buffer):
    tile_path = tmpdir / 'tiles'
    tiling.tile(ds, str(tile_path), chunks=chunks, buffer=buffer)

    if isinstance(buffer, int):
        buffer_dict = {dim: buffer for dim in ds.dims.keys()}
    else:
        buffer_dict = {dim: buffer[dim] if dim in buffer else 0
                       for dim in chunks.keys()}

    # Check whether the correct number of tiles has been created
    nchunks = np.prod([
        int(np.ceil(ds.dims[dim] / n))
        #  - np.floor(buffer_dict[dim] / n))
        for dim, n in chunks.items()
    ])
    tile_files = list(map(str, tile_path.listdir()))
    assert len(tile_files) == nchunks

    for f in tile_files:
        t = open_dataset(f)
        assert_equal(ds.attrs, t.attrs)

        for dim, val in chunks.items():
            assert t.dims[dim] <= val + 2*buffer_dict[dim]

    if buffer == 0 and len(chunks) == 1:
        mf_data = xr.open_mfdataset(tile_files, engine='h5netcdf').compute()
        assert_equal_data(ds, mf_data)

    else:
        merged = tiling.auto_merge(tile_files)
        assert_equal_data(ds, merged)


@pytest.mark.parametrize('fn', [
    lambda x: x,
    lambda x: x+1
])
def test_map_over_tiles(fn):
    # mapped = tiling.map_over_tiles(fn)
    ...
