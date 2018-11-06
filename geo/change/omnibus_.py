"""
This module contains the change detection algorithm by
Conradsen et al. (2015).

TODO: Make all functions work with xarray Datasets

"""
from ..io import disassemble_complex
from ..filter import boxcar
from . import _omnibus
from .change_ import ChangeDetection
import numpy as np
import xarray as xr
from scipy.stats import chi2
import dask.array as da


try:
    type(profile)
except NameError:
    def profile(fn): return fn


@profile
def omnibus_test(ds, n=1, axis=0):
    """Compute the test statistic to determine whether a change has occurred.

    TODO: make accept xarray.Dataset √
    TODO: properly handle all the overflow warnings
    TODO: properly handle np.nan values
    TODO: deprecate axis parameter

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain dimensions 'lat', 'lon' and 'time', as well as the data
        variables 'C11', 'C12', and 'C22'.
    Cs : numpy.array, shape (N, ..., 2, 2)
        A numpy array containing N 2x2 complex covariance matrices along
        arbitrarily many further axes.
    n : int, optional
        The number of looks (default: 1).
    axis : int, optional
        The axis along which the test statistic is to be computed
        (default: 0).

    Returns
    -------
    float, shape (...)
        The test statistic Q for testing the absence of change points.
    """
    if not isinstance(ds, xr.Dataset):
        raise ValueError("`ds` is not an instance of xarray.Dataset")

    # NOTE: For now, only support dualpol
    c11 = ds.C11
    c12 = ds.C12
    c22 = ds.C22
    c21 = np.conj(c12)
    # p: dimension (dual pol: p=2, full pol: p=3)
    p = 2
    # k: number of matrices (time steps)
    k = ds.sizes['time']
    c = da.stack([c11, c12, c21, c22], axis=-1)
    # Reshape the 4 covariance matrix entries into the (2, 2) matrix shape
    c = c.reshape(c.shape[:-1] + (2, 2))
    Xs = n * c
    X = Xs.sum(axis=0)
    # NOTE: np.linalg.det treats the last two dimensions as the square
    # matrices to compute determinants for.
    # np.linalg.det will reduce the dimension to 3:
    numerator = np.prod(np.abs(np.linalg.det(Xs)), axis=0)
    denominator = np.abs(np.linalg.det(X) ** k)
    Q = (float(k) ** (p * k) * numerator / denominator) ** n

    # probability
    # f, rho, omega2 taken from Conradsen et al. 2015
    f = (k - 1) * p**2
    rho = 1 - (2 * p**2 - 1) / (6 * (k - 1) * p) * (k/n - 1/(n*k))
    omega2 = p**2 * (p**2 - 1) / (24 * rho**2) \
        * (k/(n**2) - 1/((n*k)**2)) \
        - p**2 * (k - 1) / 4 * (1 - 1/rho)**2

    z = -2 * rho * np.log(Q)
    _P1 = chi2.cdf(z, f)
    _P2 = chi2.cdf(z, f+4)
    prob = _P1 + omega2 * (_P2 - _P1)
    return prob


def omnibus(ds, n=1):
    """Wrapper for the Cython function `_change.array_omnibus`
    """
    values = np.stack([
        ds.C11, ds.C12.real, ds.C12.imag, ds.C22
    ], axis=-1)
    return _omnibus.array_omnibus(values, n=n)


def change_detection_legacy(ds, alpha=0.01, ml=None, n=1):
    """
    Implement the change detection algorithm proposed by Conradsen et al.
    (2015).

    Parameters
    ----------
    ds : xarray.Dataset
        A (multilooked) dataset in covariance matrix format.
    alpha : float (0. ... 1.), optional
        The significance level (default: 0.01).
    ml : int, optional
        Multilooking window size. If `None`, no multilooking is performed and
        the dataset is assumed to already be multilooked (default: None)
    n : int, optional
        The number of looks in `ds`. If `ml` is specified this parameter is
        ignored (default: 1).

    Returns
    -------
    xarray.Dataset
        The index of first detected change
    """
    # Multilooking
    if ml is not None:
        ds_m = boxcar(ds, w=ml)
        n = ml ** 2
    else:
        ds_m = ds

    # Need to keep track of the changes found or rejected.
    k = ds_m.sizes['time']
    nrows = ds_m.sizes['lat']
    ncols = ds_m.sizes['lon']

    #
    # NOTE: For now, only do l=0 and find only the _first_ significant change.
    #

    # t0 = 0
    # Select ds[t0:] in the time dimension
    # subset = ds.isel(time=slice(t0, None))
    # Compute the change probability in an omnibus test for subset
    # p_H0 = omnibus_test(subset)
    # no_changes = (p_H0 < alpha)

    def first_change_after(ds, t0, alpha, n):
        """
        Compute the index of the first change
        """
        k = ds.sizes['time']
        nrows = ds_m.sizes['lat']
        ncols = ds_m.sizes['lon']
        first_change = np.full((nrows, ncols), np.nan, dtype=np.float32)

        # j is the number of time points to consider in the omnibus tests
        for j in range(2, k - t0):
            subset = ds_m.isel(time=slice(t0, t0 + j))
            p_H0 = omnibus(subset, n=n)
            # continue until first significant change
            changes = (p_H0 >= 1 - alpha)
            new_first_change = np.logical_and(changes, np.isnan(first_change))
            first_change[new_first_change] = t0 + j - 1

        return first_change
        # Create a new dataset to be returned, delete all variables

        # change_ds = ds_m.copy()
        # for var in ds_m.data_vars:
        #     del change_ds[var]
        # change_ds['first_change'] = (('lat', 'lon'), first_change)

        # return change_ds

    # Iteratively set the start index, but skip the last one (t0 == k - 1),
    # as the changepoint detection make no sense for a single time step.
    changes = np.empty((k - 1, nrows, ncols), dtype=np.float32)
    for t0 in range(k - 1):
        first_change = first_change_after(ds_m, t0=t0, alpha=alpha, n=n)
        changes[t0, :, :] = first_change

    # Determine the changes
    changes_bool = _omnibus.change_array_to_bool(changes).astype(bool)

    # return changes_bool

    # Create a dataset of all the change points.
    change_ds = ds_m.copy()
    for var in ds_m.data_vars:
        del change_ds[var]
    change_ds['time'] = ds_m['time'][:-1]
    change_ds['changes'] = (('time', 'lat', 'lon'), changes_bool)

    return change_ds


def _change_detection(ds, alpha=0.01, ml=None, n=1, njobs=1):
    """
    Implement the change detection algorithm proposed by Conradsen et al.
    (2015).

    Parameters
    ----------
    ds : xarray.Dataset
        A (multilooked) dataset in covariance matrix format.
    alpha : float (0. ... 1.), optional
        The significance level (default: 0.01).
    ml : int, optional
        Multilooking window size. If `None`, no multilooking is performed and
        the dataset is assumed to already be multilooked (default: None)
    n : int, optional
        The number of looks in `ds`. If `ml` is specified this parameter is
        ignored (default: 1).

    Returns
    -------
    xarray.DataArray
        A boolean DataArray indicating whether a change occurred at each
        (lat, lon, time) coordinate.
    """
    ds.persist()

    ds_m = disassemble_complex(ds)

    # Multilooking
    if ml is not None:
        ds_m = boxcar(ds_m, w=ml)
        n = ml ** 2

    values = ds_m[['C11', 'C12__re', 'C12__im', 'C22']].to_array() \
        .transpose('lat', 'lon', 'time', 'variable').values

    change = _omnibus.change_detection(values, alpha=alpha, n=n, njobs=njobs)

    coords = ds.transpose('lat', 'lon', 'time').coords
    change_arr = xr.DataArray(np.asarray(change, dtype=bool),
                              dims=coords.keys(), coords=coords,
                              attrs=ds.attrs, name='change')

    return change_arr


class OmnibusTest(ChangeDetection):

    def __init__(self, ml=None, n=1, *args, **kwargs):
        self.ml = ml
        self.n = n
        super().__init__(*args, **kwargs)

    def detect(self, ds, alpha=0.01):
        return _change_detection(ds, alpha=alpha, ml=self.ml, n=self.n,
                                 njobs=self.njobs)
