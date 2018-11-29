import numpy as np
from ..utils import get_vars_for_dims, expand_variables
from ._nlmeans import _pixelwise_nlmeans_3d


def nlmeans(ds, r, sigma, h, f=1, **kwargs):
    """
    Non-Local Means (Buades2011).

    Buades, A., Coll, B., & Morel, J.-M. (2011). Non-Local Means Denoising.
    Image Processing On Line, 1, 208–212.
    https://doi.org/10.5201/ipol.2011.bcm_nlm

    Parameters
    ----------
    ds : xarray.Dataset
    r : dict
        e.g. {'lat': 3, 'lon': 3, 'time': 1}
    sigma : float
    h : float
    dims : tuple
    f : int

    """
    orig_dims = tuple(ds.dims)
    dims = tuple(r.keys())

    r_ = np.array(list(r.values()), dtype=np.uint32)
    f_ = np.array([f if _ > 0 else 0 for _ in r_], dtype=np.uint32)

    variables = get_vars_for_dims(ds, dims)
    other_variables = get_vars_for_dims(ds, dims, invert=True)

    ordered_dims = dims + tuple(set(orig_dims) - set(dims)) + ('variable',)

    # convert to DataArray
    da_ordered = ds[variables].to_array().transpose(*ordered_dims)
    da_filtered = da_ordered.copy()

    # Pad dimensions to 3D (4D with variables)
    # Extra dimensions are prepended
    arr = np.array(da_ordered.values, ndmin=4, copy=False)
    output = np.array(da_filtered.values, ndmin=4, copy=False)
    pad = np.zeros(3 - len(r_), dtype=r_.dtype)
    r_ = np.concatenate([pad, r_])
    f_ = np.concatenate([pad, f_])

    _pixelwise_nlmeans_3d(arr, output, r_, f_, sigma, h)

    result = expand_variables(da_filtered).transpose(*orig_dims)

    for v in other_variables:
        result[v] = ds[v]

    return result
