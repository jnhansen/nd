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
