import pytest
import inspect
import xarray as xr
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_allclose as xr_assert_allclose
from xarray.testing import assert_identical as xr_assert_identical
import nd.filters
from nd.testing import all_algorithms
from nd.filters import Filter
from nd.testing import generate_test_dataset


ds = generate_test_dataset(nlat=20, nlon=30, ntime=10)
filters = [alg for alg in all_algorithms(nd.filters)
           if issubclass(alg[1], Filter) and alg[0] != 'Filter']
filter_classes = [_[1] for _ in filters]


@pytest.mark.parametrize('f', filter_classes)
def test_filter_input_output(f):
    instance = f(
        dims=('lat', 'lon')
    )
    result = instance.apply(ds)

    # Check that the output is an xarray Dataset
    assert isinstance(result, xr.Dataset)

    # Check that the variable dimensions are unchanged.
    for v in ds.data_vars:
        assert_equal(ds[v].dims, result[v].dims)
        assert_equal(ds[v].shape, result[v].shape)


@pytest.mark.parametrize('f', filter_classes)
def test_filter_signature(f):
    sig = inspect.signature(f._filter)
    params = list(sig.parameters.keys())
    assert_equal(params, ['self', 'arr', 'axes', 'output'])


@pytest.mark.parametrize('f', filter_classes)
def test_filter_mutable_dimension(f):
    # Check that changing the order of dimensions doesn't change
    # the outcome up to numeric precision.
    xr_assert_allclose(
        f(dims=('lat', 'lon')).apply(ds),
        f(dims=('lon', 'lat')).apply(ds)
    )
