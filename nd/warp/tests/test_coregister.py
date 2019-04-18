import pytest
from nd.testing import generate_test_dataset, generate_test_dataarray
from nd.utils import get_vars_for_dims
from nd.warp.coregister_ import Coregistration
from nd.filters import GaussianFilter
import numpy as np
import xarray as xr
import skimage.transform
import skimage.feature


def create_misaligned_dataset(**kwargs):
    np.random.seed(0)
    ds = generate_test_dataset(**kwargs)
    datavars = get_vars_for_dims(ds, ['time', 'x', 'y'])
    # Create some structure from t=0
    d0 = GaussianFilter(sigma=3).apply(ds.isel(time=0))
    d0 = d0/d0.max()
    # Generate a noisy dataset based on the structure
    ds = ds/ds.max()
    ds = ds + d0
    # Introduce some shifts
    ntime = ds.dims['time']
    shifts = np.random.rand(ntime, 2)
    shifts[0, :] = 0
    for t in range(1, ntime):
        src = ds.isel(time=t)
        transf = skimage.transform.AffineTransform(translation=shifts[t, :])
        # Apply transform to each variable
        for v in datavars:
            ds[v].loc[dict(time=ds['time'][t])] = \
                skimage.transform.warp(src[v].values, transf, order=3)

    return ds, shifts


def check_shifts(ds):
    ref_var = 'C11'
    ref = ds.isel(time=0)[ref_var].values
    shifts = []
    for t in range(1, ds.dims['time']):
        # Estimate shift
        shift = skimage.feature.register_translation(
            ds.isel(time=t)[ref_var].values, ref, upsample_factor=30)[0]
        shifts.append(shift)
    return np.array(shifts)


def test_coregistration():
    ds, shifts = create_misaligned_dataset(nx=100, ny=100)
    cor = Coregistration(upsampling=30)
    ds_cor = cor.apply(ds)
    shifts = check_shifts(ds_cor)
    print(shifts)
    assert (np.abs(shifts) <= 0.1).all()