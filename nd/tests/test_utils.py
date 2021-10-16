import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_raises_regex
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_identical as xr_assert_identical
from xarray.testing import assert_allclose as xr_assert_allclose
from nd import utils
from nd.io import assemble_complex, disassemble_complex
from nd.testing import (equal_list_of_dicts, generate_test_dataset,
                        generate_test_dataarray)
import time
from datetime import datetime
from dateutil.tz import tzutc
from collections import OrderedDict


@pytest.mark.parametrize('fmt', [
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y/%m/%dT%H:%M:%S.%fZ'
])
@pytest.mark.parametrize('tz', [
    True, False
])
def test_str2date(fmt, tz):
    tzinfo = tzutc() if tz else None
    d = datetime(2018, 1, 6, 12, 34, 56, tzinfo=tzinfo)
    dstr = d.strftime(fmt)
    assert_equal(utils.str2date(dstr, tz=tz), d)
    assert_equal(utils.str2date(dstr, fmt=fmt, tz=tz), d)


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


def test_array_chunks():
    arr = np.random.rand(30, 30, 30)
    for axis in range(arr.ndim):
        for chunk in utils.array_chunks(arr, 5, axis=axis):
            assert chunk.shape[axis] == 5
        merged = np.concatenate(
            [chunk for chunk in utils.array_chunks(arr, 5, axis=axis)],
            axis=axis
        )
        assert_equal(arr, merged)


def test_array_chunks_return_indices():
    arr = np.random.rand(30, 30, 30)
    for axis in range(arr.ndim):
        for idx, chunk in utils.array_chunks(
                arr, 8, axis=axis, return_indices=True):
            assert_equal(chunk, arr[tuple(idx)])


def test_array_chunks_invalid_axis():
    arr = np.random.rand(50, 50)
    with assert_raises_regex(ValueError, 'out of range'):
        _ = list(utils.array_chunks(arr, 5, axis=2))


def test_block_split():
    arr = np.random.rand(30, 30, 30)
    nblocks = (1, 2, 3)
    blocks = utils.block_split(arr, nblocks)
    assert len(blocks) == np.prod(nblocks)


@pytest.mark.parametrize('nblocks', [
    (2,), (2, 2, 2)
])
def test_block_split_invalid_blocks(nblocks):
    arr = np.arange(16).reshape((4, 4))
    with assert_raises(ValueError):
        _ = utils.block_split(arr, nblocks)


def test_block_merge():
    arr = np.arange(16).reshape((4, 4))
    blocks = [
        arr[0:2, 0:2],
        arr[0:2, 2:4],
        arr[2:4, 0:2],
        arr[2:4, 2:4]
    ]
    merged = utils.block_merge(blocks, (2, 2))
    assert_equal(merged, arr)


@pytest.mark.parametrize('nblocks', [
    (2,), (2, 2, 2)
])
def test_block_merge_invalid_blocks(nblocks):
    arr = np.arange(16).reshape((4, 4))
    blocks = [
        arr[0:2, 0:2],
        arr[0:2, 2:4],
        arr[2:4, 0:2],
        arr[2:4, 2:4]
    ]
    with assert_raises(ValueError):
        _ = utils.block_merge(blocks, nblocks)


@pytest.mark.parametrize('nblocks', [
    (1, 1, 1),
    (1, 2, 3),
    (7, 6, 5)
])
def test_block_split_and_merge(nblocks):
    arr = np.random.rand(30, 30, 30)
    blocks = utils.block_split(arr, nblocks)
    merged = utils.block_merge(blocks, nblocks)
    assert_equal(arr, merged)


def _parallel_fn(ds):
    # Simulate some work with O(N) ...
    size = ds.count().to_array().values.sum()
    time.sleep(size / 20000.)
    return ds + 1


@pytest.mark.parametrize('dim', ['x', 'y', 'time'])
@pytest.mark.parametrize('chunks', [1, 2, 3])
@pytest.mark.parametrize('buffer', [0, 1, 5])
def test_xr_split_and_merge(dim, chunks, buffer):
    dims = dict(x=50, y=50, time=50)
    ds = generate_test_dataset(dims=dims)
    parts = list(utils.xr_split(ds, dim=dim, chunks=chunks, buffer=buffer))
    merged = utils.xr_merge(parts, dim=dim, buffer=buffer)
    xr_assert_equal(ds, merged)


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


def test_parallel_invalid_dim():
    dims = dict(x=50, y=50, time=50)
    ds = generate_test_dataset(dims=dims)

    def _fn(ds):
        return (ds - ds.mean('time')) / ds.std('time')

    with assert_raises_regex(
            ValueError, "The dataset has no dimension 'invalid'"):
        _ = utils.parallel(_fn, chunks=4, dim='invalid')(ds)


def test_parallel_merge():
    dims = dict(x=50, y=50, time=50)
    ds = generate_test_dataset(dims=dims)

    def _fn(ds):
        return (ds - ds.mean('time')) / ds.std('time')

    result = _fn(ds)
    result_1 = utils.parallel(_fn, 'x')(ds)
    result_2 = utils.xr_merge(
        utils.parallel(_fn, 'x', merge=False)(ds), dim='x')

    xr_assert_allclose(result, result_1)
    xr_assert_allclose(result_1, result_2)


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


def test_parse_docstrings():
    doc = """
    Preamble

    More information

    Parameters
    ----------
    a : int
    b : bool
        More information

    Returns
    -------
    int
        The result

    """
    parsed = utils.parse_docstring(doc)
    assembled = utils.assemble_docstring(parsed)
    stripped_doc = '\n'.join([l.rstrip() for l in doc.split('\n')])
    assert_equal(stripped_doc, assembled)


def test_is_complex():
    ds = assemble_complex(generate_test_dataset())
    # Check Dataset
    assert utils.is_complex(ds)
    assert not utils.is_complex(disassemble_complex(ds))
    # Check DataArray
    assert utils.is_complex(ds.C12)
    assert not utils.is_complex(ds.C11)


def test_is_complex_invalid_input():
    with assert_raises_regex(ValueError, 'Not an xarray Dataset or DataArray'):
        utils.is_complex('a string')


@pytest.mark.parametrize('dims', [
    ('x', 'y'), ('time',)
])
def test_apply(dims):
    ds = generate_test_dataset()
    ref = ds.mean(dims)
    result = utils.apply(
        ds, np.mean, signature='({})->()'.format(",".join(dims))
    )
    xr_assert_allclose(
        result.transpose(*ref.nd.dims),
        ref.transpose(*ref.nd.dims)
    )


def test_apply_with_vars():
    ds = generate_test_dataset()
    ref = ds.to_array(dim='var').mean('var')
    result = utils.apply(
        ds, lambda a: a.mean(axis=1), signature='(time,var)->(time)'
    )
    xr_assert_allclose(
        result.transpose(*ref.nd.dims),
        ref.transpose(*ref.nd.dims)
    )


def test_apply_with_vars_keep_vars():
    ds = generate_test_dataset()
    ref = ds.mean('time')
    result = utils.apply(
        ds, lambda a: a.mean(axis=0), signature='(time,var)->(var)'
    )
    xr_assert_allclose(
        result.transpose(*ref.nd.dims),
        ref.transpose(*ref.nd.dims)
    )


@pytest.mark.parametrize('args,kwargs', [
    ((1, 2, 3), dict(c=4, d=5)),
    ((1,), dict(b=2, d=3)),
    ((1, 2, 3, 4, 5), dict()),
    ((), dict(b=2, a=1)),
])
def test_extract_arguments(args, kwargs):
    def fn(a, b, *args, c=None, **kwargs):
        return OrderedDict(
            a=a, b=b, args=args, c=c, kwargs=kwargs
        )

    bound = utils.extract_arguments(fn, args, kwargs)
    actual = fn(*args, **kwargs)
    assert_equal(bound, actual)


@pytest.mark.parametrize('req,exists', [
    ('numpy', True),
    ('made_up_module', False),
    ('gdal', True),
    (['numpy', 'xarray'], True),
    (['numpy', 'made_up_module'], False),
])
def test_check_requirements(req, exists):
    assert utils.check_requirements(req) == exists


@pytest.mark.parametrize('req,exists', [
    ('numpy', True),
    ('made_up_module', False),
    ('gdal', True),
    (['numpy', 'xarray'], True),
    (['numpy', 'made_up_module'], False),
])
def test_requires_class(req, exists):
    # Define class with given requirements
    @utils.requires(req)
    class C():
        def __init__(self):
            pass

    if exists:
        # Should be able to instantiate class
        try:
            C()
        except Exception as e:
            pytest.fail(e)
    else:
        # Class instantiation should fail
        with assert_raises_regex(ImportError, f'requires .* {req}'):
            C()


@pytest.mark.parametrize('req,exists', [
    ('numpy', True),
    ('made_up_module', False),
    ('gdal', True),
    (['numpy', 'xarray'], True),
    (['numpy', 'made_up_module'], False),
])
def test_requires_function(req, exists):
    # Define class with given requirements
    @utils.requires(req)
    def func():
        pass

    if exists:
        # Should be able to instantiate class
        try:
            func()
        except Exception as e:
            pytest.fail(e)
    else:
        # Class instantiation should fail
        with assert_raises_regex(ImportError, f'requires .* {req}'):
            func()


def test_squeeze():
    a = generate_test_dataarray(dims={'y': 1})
    value = a.values[0]
    squeezed = utils.squeeze(a)
    assert isinstance(squeezed, float)
    assert value == squeezed

    b = generate_test_dataarray(dims={'y': 2})
    xr_assert_identical(b, utils.squeeze(b))
