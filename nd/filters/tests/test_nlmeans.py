from nd.filters import nlmeans
from nd.testing import generate_test_dataset, assert_all_true
import numpy as np


ds = generate_test_dataset(nlat=20, nlon=20, ntime=10)


def test_nlmeans():
    ds_nlm = nlmeans(ds, r={}, f=1, sigma=2, h=2)
    # Check that the difference between the original data and the filtered
    # data is not too large.
    assert (np.abs(ds.mean() - ds_nlm.mean()).to_array() < 1e-3).all()


def test_nlmeans_zero_radius():
    # Check that passing r={} returns the input Dataset unaltered.
    ds_nlm = nlmeans(ds, r={}, f=1, sigma=1, h=1)
    assert ds.equals(ds_nlm)


def test_reduce_std():
    # Check that applying nlmeans reduces the standard deviation.
    ds_nlm = nlmeans(ds, r={'lat': 1, 'lon': 1, 'time': 0}, sigma=2, h=2)
    assert_all_true(ds_nlm.std() < ds.std())


def test_ignore_time_dimension():
    # Each timestep should be treated completely
    # independent from each other if r['time'] == 0.
    ds_nlm = nlmeans(ds, r={'lat': 1, 'lon': 1, 'time': 0}, sigma=2, h=2)
    t0 = ds.isel(time=0)
    t0_nlm = nlmeans(t0, r={'lat': 1, 'lon': 1}, sigma=2, h=2)
    assert (np.abs((ds_nlm.isel(time=0) - t0_nlm).to_array()) < 1e-8).all()
