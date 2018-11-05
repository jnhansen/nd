import numpy as np
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical
from geo import utils
from geo.utils.testing import equal_list_of_dicts, generate_test_dataset
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
    assert serial_time > parallel_time
