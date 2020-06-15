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


ds = generate_test_dataset(dims={'y': 20, 'x': 30, 'time': 10})
filters = [alg for alg in all_algorithms(nd.filters)
           if issubclass(alg[1], Filter) and alg[0] != 'Filter']
filter_classes = [_[1] for _ in filters]


@pytest.mark.parametrize('f', filter_classes)
def test_filter_input_output(f):
    instance = f(
        dims=('y', 'x')
    )
    result = instance.apply(ds)

    # Check that the output is an xarray Dataset
    assert isinstance(result, xr.Dataset)

    # Check that the variable dimensions are unchanged.
    for v in ds.data_vars:
        print(ds[v].dims, result[v].dims)
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
        f(dims=('y', 'x')).apply(ds),
        f(dims=('x', 'y')).apply(ds)
    )


@pytest.mark.parametrize('f', filter_classes)
@pytest.mark.parametrize('dims', [('x', 'y'), ('x', 'y', 'time')])
def test_parallelized_filter(f, dims):
    xr_assert_allclose(
        f(dims=dims).apply(ds),
        f(dims=dims).apply(ds, njobs=2)
    )
