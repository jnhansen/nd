import pytest
import numpy as np
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical
from nd import utils
from nd.testing import equal_list_of_dicts, generate_test_dataset
import time


def test_str2date():
    pass


def test_dict_product():
    d = {'a': [1, 2, 3], 'b': [5, 6]}
    result = [{'a': 1, 'b': 5},
              {'a': 1, 'b': 6},
              {'a': 2, 'b': 5},
              {'a': 2, 'b': 6},
              {'a': 3, 'b': 5},
              {'a': 3, 'b': 6}]
    assert equal_list_of_dicts(list(utils.dict_product(d)), result)


def test_chunks():
    ll = np.arange(100)
    n = 15
    for i, c in enumerate(utils.chunks(ll, n)):
        assert_equal(c, ll[i*n:(i+1)*n])


def _parallel_fn(ds):
    # Simulate some work with O(N) ...
    size = ds.count().to_array().values.sum()
    time.sleep(size / 20000.)
    return ds + 1


def test_parallel():
    ds = generate_test_dataset()
    t = time.time()
    result_serial = _parallel_fn(ds)
    serial_time = time.time() - t
    t = time.time()
    result_parallel = utils.parallel(_parallel_fn, chunks=4)(ds)
    parallel_time = time.time() - t
    # Assert that the results are identical
    xr_assert_identical(result_serial, result_parallel)

    # Assert that the parallel execution was more than three times faster
    import multiprocessing as mp
    if mp.cpu_count() <= 1:
        pytest.skip("Execution time is only faster if there are "
                    "multiple cores")
    assert serial_time > parallel_time


def test_select_list():
    complete = [{'a': 1, 'b': 2}, {'a': 2, 'b': 2}, {'a': 1, 'b': 1}]
    expected = [{'a': 1, 'b': 2}, {'a': 1, 'b': 1}]
    selected = utils.select(complete, lambda o: o['a'] == 1)
    assert_equal(expected, selected)


def test_select_dict():
    complete = {'x': {'a': 1, 'b': 2},
                'y': {'a': 2, 'b': 2},
                'z': {'a': 1, 'b': 1}}
    expected = {'x': {'a': 1, 'b': 2}, 'z': {'a': 1, 'b': 1}}
    selected = utils.select(complete, lambda o: o['a'] == 1)
    assert_equal(expected, selected)


def test_select_list_first():
    complete = [{'a': 1, 'b': 2}, {'a': 2, 'b': 2}, {'a': 1, 'b': 1}]
    assert_equal(utils.select(complete, lambda o: o['a'] == 1, first=True),
                 {'a': 1, 'b': 2})
    assert_equal(utils.select(complete, lambda o: 'c' in o, first=True),
                 None)


def test_select_dict_first():
    complete = {'x': {'a': 1, 'b': 2},
                'y': {'a': 2, 'b': 2},
                'z': {'a': 1, 'b': 1}}
    assert utils.select(complete, lambda o: o['a'] == 1, first=True) \
        in [{'a': 1, 'b': 2}, {'a': 1, 'b': 1}]


def test_get_vars_for_dims():
    ds = generate_test_dataset(var=['var1', 'var2'])
    ds['other'] = 1
    ds['spatial'] = (('y', 'x'),
                     np.ones((ds.dims['y'], ds.dims['x'])))
    all_vars = {'var1', 'var2', 'other', 'spatial'}

    for dims, variables in [
        ([], all_vars),
        (['y', 'x'], {'var1', 'var2', 'spatial'}),
        (['y', 'x', 'time'], {'var1', 'var2'})
    ]:
        assert_equal(
            set(utils.get_vars_for_dims(ds, dims)),
            variables
        )
        assert_equal(
            set(utils.get_vars_for_dims(ds, dims, invert=True)),
            all_vars - variables
        )


def test_expand_variables():
    ds = generate_test_dataset()
    da = ds.to_array(dim='new_dim')
    ds_new = utils.expand_variables(da, dim='new_dim')
    xr_assert_identical(ds, ds_new)
