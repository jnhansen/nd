import numpy as np
import scipy.ndimage.filters as snf


def _convolve(arr, kernel, out=None, **kwargs):
    """
    Low-level convolution function.
    """
    print(arr.shape, kernel.shape)
    return snf.convolve(arr, kernel, **kwargs)
    # return cv2.filter2D(arr, ddepth=-1, kernel=kernel, anchor=(-1, -1),
    #                     delta=0, borderType=cv2.BORDER_REFLECT)


def _expand_kernel(kernel, kernel_dims, new_dims):
    """
    Reshape a kernel spanning some dimensions to cover a superset of
    dimensions.

    Parameters
    ----------
    kernel : ndarray
        An n-dimensional kernel.
    kernel_dims : tuple
        The dimensions corresponding to the kernel axes.
    new_dims : tuple
        The dimensions of the dataset to which the kernel needs to be applied.
        Must be a superset of `kernel_dims`.

    Returns
    -------
    ndarray
        The reshaped kernel.

    Raises
    ------
    ValueError
        Will raise a ValueError if the dimensions of the kernel don't match
        `kernel_dims`.
    ValueError
        Will raise a ValueError if `new_dims` is not a superset of
        `kernel_dims`.
    """

    if not set(new_dims).issuperset(set(kernel_dims)):
        raise ValueError('`new_dims` must be a superset of `kernel_dims`.')

    if kernel.ndim != len(kernel_dims):
        raise ValueError('The length of `kernel_dims` must match the '
                         'dimension of `kernel`.')

    new_kernel_shape = np.ones(len(new_dims), dtype=int)
    new_kernel_shape[[new_dims.index(_) for _ in kernel_dims]] = kernel.shape
    return kernel.reshape(new_kernel_shape)


# TODO: offer `inplace=True` option?
def convolve(ds, kernel, dims=('lat', 'lon'), **kwargs):
    """
    Kernel-convolution of an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    kernel : ndarray
        The convolution kernel.
    dims : tuple, optional
        The dataset dimensions corresponding to the kernel axes
        (default: ('lat', 'lon')). The length of the tuple must match the
        number of dimensions of the kernel.
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.convolve``.

    Returns
    -------
    xarray.Dataset
        The convolved dataset.
    """

    # Make sure the dimensions we want to filter are first.
    # dims = ('lat', 'lon')
    ordered_dims = tuple(dims) + tuple(set(ds.dims) - set(dims))

    # Create new dataset (allocate memory).
    ds_ordered = ds.transpose(*ordered_dims)
    ds_conv = ds_ordered.copy()

    # Apply filter to every variable that contains all dimensions
    # given in `dim`.
    for v in ds_conv.data_vars:
        # Skip variables that do not contain all specified dimensions.
        vdims = ds_conv[v].dims
        if not set(vdims).issuperset(set(dims)):
            continue

        # Reshape kernel to cover all variable dimensions.
        nd_kernel = _expand_kernel(kernel, dims, vdims)

        values = ds_conv[v].values

        if np.iscomplexobj(values):
            re_conv = _convolve(np.real(values), nd_kernel, **kwargs)
            im_conv = _convolve(np.imag(values), nd_kernel, **kwargs)
            v_convolved = re_conv + 1j * im_conv
        else:
            v_convolved = _convolve(values, nd_kernel, **kwargs)

        ds_conv[v] = (ds_conv[v].dims, v_convolved)

    return ds_conv


def boxcar(ds, w, dims=('lat', 'lon'), **kwargs):
    """
    A boxcar filter.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    w : int
        The width of the boxcar window. Should be an odd integer in order to
        ensure symmetry.
    dims : tuple of str
        The dimensions along which to apply the filter
        (default: ('lat', 'lon')).
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.convolve``.

    Returns
    -------
    xarray.Dataset
        The filtered dataset.
    """

    N = len(dims)
    kernel = np.ones((w,) * N, dtype=np.float64) / w**N
    return convolve(ds, kernel=kernel, dims=dims, **kwargs)


def gaussian(ds, sigma, dims=('lat', 'lon'), **kwargs):
    """
    A Gaussian filter.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    sigma : float or sequence of float
        The standard deviation for the Gaussian kernel. If sequence, this is
        set individually for each dimension.
    dims : tuple of str, optioal
        The dimensions along which to apply the Gaussian filtering
        (default: ('lat', 'lon')).
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.gaussian_filter``.

    Returns
    -------
    xarray.Dataset
        The filtered dataset.
    """

    if isinstance(sigma, (int, float)):
        sigma = [sigma] * len(dims)

    # Make sure the dimensions we want to filter are first.
    # dims = ('lat', 'lon')
    ordered_dims = tuple(dims) + tuple(set(ds.dims) - set(dims))

    # Create new dataset (allocate memory).
    ds_ordered = ds.transpose(*ordered_dims)
    ds_gauss = ds_ordered.copy()

    # Apply filter to every variable that contains all dimensions
    # given in `dim`.
    for v in ds_gauss.data_vars:
        # Skip variables that do not contain all specified dimensions.
        vdims = ds_gauss[v].dims
        if not set(vdims).issuperset(set(dims)):
            continue

        # Generate n-dimensional sigma
        ndsigma = [0] * len(vdims)
        for d, s in zip(dims, sigma):
            ndsigma[vdims.index(d)] = s

        values = ds_gauss[v].values

        if np.iscomplexobj(values):
            re_gauss = snf.gaussian_filter(np.real(values), sigma=ndsigma,
                                           **kwargs)
            im_gauss = snf.gaussian_filter(np.imag(values), sigma=ndsigma,
                                           **kwargs)
            v_gauss = re_gauss + 1j * im_gauss
        else:
            v_gauss = snf.gaussian_filter(values, sigma=ndsigma, **kwargs)

        ds_gauss[v] = (ds_gauss[v].dims, v_gauss)

    return ds_gauss
