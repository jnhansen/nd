import numpy as np
import scipy.ndimage.filters as snf
from . import Filter


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


class ConvolutionFilter(Filter):
    """
    Kernel-convolution of an xarray.Dataset.

    Parameters
    ----------
    dims : tuple, optional
        The dataset dimensions corresponding to the kernel axes
        (default: ('lat', 'lon')). The length of the tuple must match the
        number of dimensions of the kernel.
    kernel : ndarray
        The convolution kernel.
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.convolve``.
    """

    per_variable = True
    kwargs = {}

    def __init__(self, dims, kernel=None, **kwargs):
        if kernel is None:
            kernel = np.ones([1] * len(dims))
        self.dims = tuple(dims)
        self.kernel = kernel
        self.kwargs = kwargs

    def _filter(self, arr, axes, output):
        # Reshape kernel to match dimension of input array.
        new_kernel_shape = np.ones(arr.ndim, dtype=int)
        new_kernel_shape[list(axes)] = self.kernel.shape
        nd_kernel = self.kernel.reshape(new_kernel_shape)
        if np.iscomplexobj(arr):
            snf.convolve(np.real(arr), nd_kernel, output=np.real(output),
                         **self.kwargs)
            snf.convolve(np.imag(arr), nd_kernel, output=np.imag(output),
                         **self.kwargs)
        else:
            snf.convolve(arr, nd_kernel, output=output, **self.kwargs)


class BoxcarFilter(ConvolutionFilter):
    """
    A boxcar filter.

    Parameters
    ----------
    dims : tuple of str
        The dimensions along which to apply the filter
        (default: ('lat', 'lon')).
    w : int
        The width of the boxcar window. Should be an odd integer in order to
        ensure symmetry.
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.convolve``.
    """

    def __init__(self, dims, w=3, **kwargs):
        N = len(dims)
        self.dims = tuple(dims)
        self.kernel = np.ones((w,) * N, dtype=np.float64) / w**N
        self.kwargs = kwargs


class GaussianFilter(Filter):
    """
    A Gaussian filter.

    Parameters
    ----------
    dims : tuple of str, optioal
        The dimensions along which to apply the Gaussian filtering
        (default: ('lat', 'lon')).
    sigma : float or sequence of float
        The standard deviation for the Gaussian kernel. If sequence, this is
        set individually for each dimension.
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.gaussian_filter``.

    Returns
    -------
    xarray.Dataset
        The filtered dataset.
    """

    def __init__(self, dims, sigma=1, **kwargs):
        if isinstance(sigma, (int, float)):
            sigma = [sigma] * len(dims)
        self.dims = tuple(dims)
        self.sigma = sigma
        self.kwargs = kwargs

    def _filter(self, arr, axes, output):
        # Generate n-dimensional sigma
        ndsigma = [0] * arr.ndim
        for ax, s in zip(axes, self.sigma):
            ndsigma[ax] = s

        if np.iscomplexobj(arr):
            snf.gaussian_filter(np.real(arr), sigma=ndsigma,
                                output=np.real(output), **self.kwargs)
            snf.gaussian_filter(np.imag(arr), sigma=ndsigma,
                                output=np.real(output), **self.kwargs)
        else:
            snf.gaussian_filter(arr, sigma=ndsigma, output=output,
                                **self.kwargs)
