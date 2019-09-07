from nd.filters import NLMeansFilter
from nd.testing import generate_test_dataset, assert_all_true
import numpy as np


ds = generate_test_dataset(dims={'y': 20, 'x': 20, 'time': 10})


def test_nlmeans():
    ds_nlm = NLMeansFilter(
        dims=('y', 'x'), r=0, f=1, sigma=2, h=2).apply(ds)
    # Check that the difference between the original data and the filtered
    # data is not too large.
    assert (np.abs(ds.mean() - ds_nlm.mean()).to_array() < 1e-3).all()


def test_nlmeans_zero_radius():
    # Check that passing r=0 returns the input Dataset unaltered.
    ds_nlm = NLMeansFilter(dims=('y', 'x'), r=0, f=1, sigma=1, h=1).apply(ds)
    assert ds.equals(ds_nlm)


def test_nlmeans_empty_dim():
    ds_nlm = NLMeansFilter(dims=(), r=1, f=1, sigma=1, h=1).apply(ds)
    assert ds.equals(ds_nlm)


def test_reduce_std():
    # Check that applying nlmeans reduces the standard deviation.
    ds_nlm = NLMeansFilter(
        dims=('y', 'x', 'time'), r=(1, 1, 0), sigma=2, h=2).apply(ds)
    assert_all_true(ds_nlm.std() < ds.std())


def test_ignore_time_dimension():
    # Each timestep should be treated completely
    # independent from each other if r['time'] == 0.
    ds_nlm = NLMeansFilter(
        dims=('y', 'x', 'time'), r=(1, 1, 0), sigma=2, h=2).apply(ds)
    t0 = ds.isel(time=0)
    t0_nlm = NLMeansFilter(
        dims=('y', 'x'), r=1, sigma=2, h=2).apply(t0)
    assert (np.abs((ds_nlm.isel(time=0) - t0_nlm).to_array()) < 1e-8).all()
