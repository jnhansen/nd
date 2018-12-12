import pytest
from nd.io.convert_ import (assemble_complex, disassemble_complex,
                            generate_covariance_matrix)
from nd.testing import generate_test_dataset, generate_test_dataarray
from xarray.testing import assert_identical as xr_assert_identical
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_allclose as xr_assert_allclose
from numpy.testing import assert_equal, assert_allclose
import numpy as np


def test_disassemble_complex_dataset():
    # Create complex dataset
    ds = generate_test_dataset(var=['b', 'c'])
    dims = tuple(ds.dims.keys())
    shape = tuple(ds.dims.values())
    complex_data = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    ds['a'] = (dims, complex_data)
    # Check that disassembling into reals works as expected
    ds_real = disassemble_complex(ds)
    assert_equal(set(ds_real.data_vars), {'a__re', 'a__im', 'b', 'c'})
    xr_assert_equal(ds['a'].real, ds_real['a__re'])
    xr_assert_equal(ds['a'].imag, ds_real['a__im'])


def test_disassemble_complex_dataarray():
    # Create complex dataset
    da = generate_test_dataarray(name='data')
    complex_data = np.random.rand(*da.shape) + 1j * np.random.rand(*da.shape)
    da.values = complex_data
    ds_real = disassemble_complex(da)
    assert_equal(set(ds_real.data_vars), {'data__re', 'data__im'})
    xr_assert_equal(da.real, ds_real['data__re'])
    xr_assert_equal(da.imag, ds_real['data__im'])


def test_assemble_complex_dataset():
    # Create real dataset with real and imag part
    ds = generate_test_dataset(var=['a__im', 'a__re', 'b', 'c'])
    # Check that assembling into complex works
    ds_complex = assemble_complex(ds)
    assert_equal(set(ds_complex.data_vars), {'a', 'b', 'c'})
    xr_assert_equal(ds_complex['a'].real, ds['a__re'])
    xr_assert_equal(ds_complex['a'].imag, ds['a__im'])


def test_assemble_and_dissassemble_complex():
    ds_orig = generate_test_dataset(var=['a__im', 'a__re', 'b', 'c'])
    ds_complex = assemble_complex(ds_orig)
    ds_real = disassemble_complex(ds_complex)
    xr_assert_identical(ds_orig, ds_real)


def test_add_time():
    pass


def test_dualpol_to_complex():
    pass


def test_generate_covariance_matrix():
    ds = generate_test_dataset(var=['VV__im', 'VV__re', 'VH__im', 'VH__re'])
    assemble_complex(ds, inplace=True)
    cov = generate_covariance_matrix(ds)
    xr_assert_allclose(np.abs(ds.VV)**2, cov.C11)
    xr_assert_allclose(np.abs(ds.VH)**2, cov.C22)
    xr_assert_allclose(ds.VV * np.conj(ds.VH), cov.C12)


def test_compact_to_complex():
    pass
