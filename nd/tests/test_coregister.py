import skimage.transform
import skimage.registration
from nd.testing import generate_test_dataset
from nd.utils import get_vars_for_dims
from nd.warp import Coregistration
from nd.filters import GaussianFilter
import numpy as np


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

    return ds, shifts[1:, :]


def check_shifts(ds):
    ref_var = 'C11'
    ref = ds.isel(time=0)[ref_var].values
    shifts = []
    for t in range(1, ds.dims['time']):
        # Estimate shift
        shift = skimage.registration.phase_cross_correlation(
            ds.isel(time=t)[ref_var].values, ref, upsample_factor=30)[0]
        shifts.append(shift)
    return np.array(shifts)


def test_coregistration():
    ds, old_shifts = create_misaligned_dataset(
        dims={'y': 200, 'x': 200, 'time': 50})
    cor = Coregistration(upsampling=50)
    ds_cor = cor.apply(ds)
    shifts = check_shifts(ds_cor)
    print(shifts)
    assert (np.abs(shifts) <= 0.2).all()
    #
    # New shifts may be slightly larger if the original shifts were very small
    #
    assert np.logical_or(
        np.abs(shifts) <= np.abs(old_shifts),
        np.abs(shifts) <= 0.1
    ).all()
