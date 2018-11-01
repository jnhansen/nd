import numpy as np
import cv2
from ._nlmeans import _pixelwise_nlmeans_3d
from ..utils import get_vars_for_dims, expand_variables

__all__ = ['convolve', 'boxcar', 'nlmeans']


def _convolve(arr, kernel, out=None):
    """
    Low-level convolution function.
    """
    return cv2.filter2D(arr, ddepth=-1, kernel=kernel, anchor=(-1, -1),
                        delta=0, borderType=cv2.BORDER_REFLECT)


# TODO: offer `inplace=True` option?
def convolve(ds, kernel, dims=('lat', 'lon')):
    """
    """
    # Make sure the dimensions we want to filter are first.
    # dims = ('lat', 'lon')
    ordered_dims = dims + tuple(set(ds.dims) - set(dims))

    # Create new dataset (allocate memory).
    ds_ordered = ds.transpose(*ordered_dims)
    ds_conv = ds_ordered.copy()

    # Apply filter to every variable that contains all dimensions
    # given in `dim`.
    for v in ds_conv.data_vars:
        if not set(ds_conv[v].dims).issuperset(set(dims)):
            continue

        values = ds_conv[v].values

        if np.iscomplexobj(values):
            im_conv = _convolve(np.real(values), kernel)
            re_conv = _convolve(np.imag(values), kernel)
            v_convolved = re_conv + 1j * im_conv
        else:
            v_convolved = _convolve(values, kernel)

        ds_conv[v] = (ds_conv[v].dims, v_convolved)

    return ds_conv


def boxcar(ds, w, dims=('lat', 'lon'), **kwargs):
    N = len(dims)
    kernel = np.ones((w,) * N, dtype=np.float64) / w**N
    return convolve(ds, kernel=kernel, dims=dims, **kwargs)


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
    dims = tuple(r.keys())
    r_ = np.array(list(r.values()), dtype=np.uint32)

    variables = get_vars_for_dims(ds, dims)
    other_variables = get_vars_for_dims(ds, dims, invert=True)
    ordered_dims = dims + ('variable',) + tuple(set(ds.dims) - set(dims))

    # convert to DataArray
    da_ordered = ds[variables].to_array().transpose(*ordered_dims)
    da_filtered = da_ordered.copy()

    arr = da_ordered.values
    output = da_filtered.values

    _pixelwise_nlmeans_3d(arr, output, r_, f, sigma, h)

    result = expand_variables(da_filtered)
    for v in other_variables:
        result[v] = ds[v]

    return result


# def multilook(ds, w=3):
#     """Multilook an image in covariance matrix representation.

#     Each matrix entry is multilooked separately.

#     TODO: take as argument the dimensions along which to convolve?
#     TODO: cythonize?

#     Parameters
#     ----------
#     ds : xarray.Dataset
#         The original image in covariance matrix representation.
#     w : int, optional
#         The window size (default: 3).

#     Returns
#     -------
#     xarray.Dataset
#         The multilooked image.
#     """
#     # Make sure lat and lon are first dimensions
#     ll = set(('lat', 'lon'))
#     dims = tuple(ll) + tuple(set(ds.dims) - ll)
#     ds_m = ds.copy().transpose(*dims)
#     for v in ds_m.data_vars:
#         if 'lat' not in ds_m[v].dims or 'lon' not in ds_m[v].dims:
#             continue
#         axes = tuple(ds_m[v].dims.index(_) for _ in ('lat', 'lon'))
#         ds_m[v] = (ds_m[v].dims,
#                    _multilook_array(ds_m[v], w=w, axes=axes))
#     return ds_m


# def _multilook_array(im, w=3, axes=(0, 1)):
#     """Return a multilooked image (equivalent to boxcar filter) with given
#     window size.

#     Parameters
#     ----------
#     im : numpy.array, shape (M, N)
#         The original image.
#     w : int, optional
#         The window size (default: 3).
#     axes : tuple, optional
#         The axes along which to convolve.

#     Returns
#     -------
#     numpy.array, shape (M, N)
#         The multilooked (filtered) image.
#     """
#     kernel_shape = [1] * im.ndim
#     for a in axes:
#         kernel_shape[a] = w
#     n = w ** len(axes)
#     values = np.asarray(im)
#     kernel = np.ones(kernel_shape, np.float64) / n
#     if np.iscomplexobj(values):
#         im_conv = ndfilter(np.real(values), kernel)
#         re_conv = ndfilter(np.imag(values), kernel)
#         return re_conv + 1j * im_conv
#     else:
#         return ndfilter(values, kernel)
