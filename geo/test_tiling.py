from geo.utils.testing import generate_test_dataset
from geo.utils import dict_product
from geo import tiling
from xarray.testing import assert_equal


# Synthesize test data
ds = generate_test_dataset(nlat=20, nlon=20, ntime=10)

# Generate tiles along lat, lon, and time dimension
slices = dict(lat=[slice(None, 10), slice(10, None)],
              lon=[slice(None, 10), slice(10, None)],
              time=[slice(None, 5), slice(5, None)])
parts = [ds.isel(**sl) for sl in dict_product(slices)]

# Generate tiles along lat, lon, and time dimension with
# buffer along lat and lon.
buffered_slices = dict(lat=[slice(None, 12), slice(8, None)],
                       lon=[slice(None, 11), slice(9, None)],
                       time=[slice(None, 5), slice(5, None)])
buffered_parts = [ds.isel(**sl) for sl in dict_product(buffered_slices)]


def test_auto_merge():
    assert_equal(ds, tiling.auto_merge(parts))


def test_auto_merge_with_buffer():
    assert_equal(ds, tiling.auto_merge(buffered_parts))
