"""
The main use of image filters is for noise reduction.
This module implements several such filters, all of which are designed to work
in an arbitrary number of dimensions.

.. document private functions
.. autofunction:: _expand_kernel
"""

from .algorithm import Algorithm, wrap_algorithm, parallelize
from abc import abstractmethod
from .utils import get_vars_for_dims, expand_variables, is_complex
from .io import disassemble_complex, assemble_complex
from ._filters import _pixelwise_nlmeans_3d
import numpy as np
import xarray as xr
import scipy.ndimage.filters as snf


__all__ = ['Filter',
           'ConvolutionFilter',
           'convolution',
           'BoxcarFilter',
           'boxcar',
           'GaussianFilter',
           'gaussian',
           'NLMeansFilter',
           'nlmeans',
           '_expand_kernel']


# -----------------
# UTILITY FUNCTIONS
# -----------------

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


# ----------------------
# ABSTRACT CLASS: FILTER
# ----------------------

class Filter(Algorithm):
    """
    The base class for a generic filter.

    Parameters
    ----------
    dims : tuple of str
        The dimensions along which the filter is applied.
    """

    # If per_variable is True, the filter is applied independently for
    # each variable. Otherwise, all variables may be used to determine the
    # filter weights.
    per_variable = True
    # If supports_complex is False, complex-values variables are disassembled
    # into two reals before applying the filter and reassembled afterwards.
    supports_complex = False
    dims = ()

    @abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @parallelize
    def apply(self, ds, inplace=False):
        """
        Apply the filter to the input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset
        inplace : bool, optional
            If True, overwrite the input data inplace (default: False).

        Returns
        -------
        xarray.Dataset
            The filtered dataset
        """
        if inplace:
            raise NotImplementedError('Inplace filtering is not currently '
                                      'implemented.')

        # This is not in the correct order, as ds.dims is always sorted
        # alphabetically.
        orig_dims = tuple(ds.dims)
        ordered_dims = self.dims + tuple(set(orig_dims) - set(self.dims))

        # Check if any of the variables are complex
        convert_complex = is_complex(ds) and not self.supports_complex
        if convert_complex:
            disassemble_complex(ds, inplace=True)

        #
        # Apply the actual filter
        #
        if isinstance(ds, xr.DataArray):
            # The data is a DataArray -->
            # Apply filter directly.
            result = ds.copy(deep=True)
            vdims = result.dims
            axes = tuple([vdims.index(d) for d in self.dims])
            self._filter(ds.values, axes, output=result.values)

        else:
            # The data is a Dataset.
            # Find all variables that match the given dimensions
            variables = get_vars_for_dims(ds, self.dims)
            other_variables = get_vars_for_dims(ds, self.dims, invert=True)

            if self.per_variable:
                # The data is a Dataset -->
                # Apply filter independently for each variable.
                result = ds.copy(deep=True)
                for v in variables:
                    vdims = result[v].dims
                    axes = tuple([vdims.index(d) for d in self.dims])
                    # Prepare data and output as numpy arrays
                    self._filter(ds[v].values, axes,
                                 output=result[v].values)

            else:
                # The data is a Dataset -->
                # The variables are an additional dimension.
                ordered_dims = ordered_dims + ('variable',)

                # convert to DataArray
                da_ordered = ds[variables].to_array().transpose(*ordered_dims)
                da_filtered = da_ordered.copy(deep=True)
                axes = tuple([da_ordered.dims.index(d) for d in self.dims])

                # Prepare data and output as numpy arrays
                self._filter(da_ordered.values, axes,
                             output=da_filtered.values)

                # Reassemble Dataset
                result = expand_variables(da_filtered)
                # Make sure all variable dimensions are in the original order
                for v in result.data_vars:
                    result[v] = result[v].transpose(*ds[v].dims)

                for v in other_variables:
                    result[v] = ds[v]

        # Reassemble complex variabbles if previously disassembled
        if convert_complex:
            assemble_complex(ds, inplace=True)

        return result

    @abstractmethod
    def _filter(self, arr, axes, output=None):
        """
        This method must be implemented by all derived classes.
        """
        return


# ------------------
# CONVOLUTION FILTER
# ------------------

class ConvolutionFilter(Filter):
    """
    Kernel-convolution of an xarray.Dataset.

    Parameters
    ----------
    dims : tuple, optional
        The dataset dimensions corresponding to the kernel axes
        (default: ('y', 'x')). The length of the tuple must match the
        number of dimensions of the kernel.
    kernel : ndarray
        The convolution kernel.
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.convolve``.
    """

    per_variable = True
    supports_complex = True
    kwargs = {}

    def __init__(self, dims=('y', 'x'), kernel=None, **kwargs):
        if kernel is None:
            kernel = np.ones([1] * len(dims))
        self.dims = tuple(dims)
        self.kernel = kernel
        self.kwargs = kwargs

    def _parallel_dimension(self, ds):
        """
        If there are dimensions that are not part of the filter,
        parallelize along the largest of those dimensions.
        """
        extra_dims = list(set(ds.nd.dims) - set(self.dims))
        if len(extra_dims) > 0:
            return sorted(
                extra_dims, key=lambda d: ds[d].size, reverse=True)[0]
        else:
            return sorted(
                ds.nd.dims, key=lambda d: ds[d].size, reverse=True)[0]

    def _buffer(self, dim):
        """
        Given the dimension to parallelize, return the required buffer.
        """
        if dim not in self.dims:
            return 0
        else:
            axis = self.dims.index(dim)
            return self.kernel.shape[axis] // 2

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


convolution = wrap_algorithm(ConvolutionFilter, 'convolution')


# -------------
# BOXCAR FILTER
# -------------

class BoxcarFilter(ConvolutionFilter):
    """
    A boxcar filter.

    Parameters
    ----------
    dims : tuple of str, optional
        The dimensions along which to apply the filter
        (default: ('y', 'x')).
    w : int
        The width of the boxcar window. Should be an odd integer in order to
        ensure symmetry.
    kwargs : dict, optional
        Extra keyword arguments passed on to
        ``scipy.ndimage.filters.convolve``.
    """

    def __init__(self, dims=('y', 'x'), w=3, **kwargs):
        N = len(dims)
        self.dims = tuple(dims)
        self.kernel = np.ones((w,) * N, dtype=np.float64) / w**N
        self.kwargs = kwargs


boxcar = wrap_algorithm(BoxcarFilter, 'boxcar')


# ------------------
# GAUSSIAN FILTER
# ------------------

class GaussianFilter(Filter):
    """
    A Gaussian filter.

    Parameters
    ----------
    dims : tuple of str, optional
        The dimensions along which to apply the Gaussian filtering
        (default: ('y', 'x')).
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

    def __init__(self, dims=('y', 'x'), sigma=1, **kwargs):
        if isinstance(sigma, (int, float)):
            sigma = [sigma] * len(dims)
        self.dims = tuple(dims)
        self.sigma = sigma
        self.kwargs = kwargs

    def _parallel_dimension(self, ds):
        """
        If there are dimensions that are not part of the filter,
        parallelize along the largest of those dimensions.
        """
        extra_dims = list(set(ds.nd.dims) - set(self.dims))
        if len(extra_dims) > 0:
            return sorted(
                extra_dims, key=lambda d: ds[d].size, reverse=True)[0]
        else:
            return sorted(
                ds.nd.dims, key=lambda d: ds[d].size, reverse=True)[0]

    def _buffer(self, dim):
        """
        Given the dimension to parallelize, return the required buffer.
        """
        if dim not in self.dims:
            return 0
        else:
            # The kernel size is determined in scipy
            # by truncation after N sigma (default 4)
            axis = self.dims.index(dim)
            sigma = self.sigma[axis]
            truncate = 4.0
            radius = int(truncate * sigma + 0.5)
            return radius

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


gaussian = wrap_algorithm(GaussianFilter, 'gaussian')


# ----------------------
# NON-LOCAL MEANS FILTER
# ----------------------

class NLMeansFilter(Filter):
    """
    Non-Local Means (Buades2011).

    Buades, A., Coll, B., & Morel, J.-M. (2011). Non-Local Means Denoising.
    Image Processing On Line, 1, 208–212.
    https://doi.org/10.5201/ipol.2011.bcm_nlm

    Parameters
    ----------
    dims : tuple of str
        The dataset dimensions along which to filter.
    r : {int, sequence}
        The radius
    sigma : float
        The standard deviation of the noise present in the data.
    h : float
    f : int
    n_eff : float, optional
        The desired effective sample size. If given, must be greater than 1
        and should be no larger than about half the pixels in the window.
        -1 means no fixed effective sample size (default: -1).
    """

    per_variable = False

    def __init__(self, dims=('y', 'x'), r=1, sigma=1, h=1, f=1, n_eff=-1):
        if isinstance(r, (int, float)):
            r = [r] * len(dims)
        self.dims = tuple(dims)
        self.r = np.array(r, dtype=np.uint32)
        self.f = np.array([f if _ > 0 else 0 for _ in self.r], dtype=np.uint32)
        self.sigma = sigma
        self.h = h
        self.n_eff = n_eff

    def _parallel_dimension(self, ds):
        """
        If there are dimensions that are not part of the filter,
        parallelize along the largest of those dimensions.
        """
        extra_dims = list(set(ds.nd.dims) - set(self.dims))
        if len(extra_dims) > 0:
            return sorted(
                extra_dims, key=lambda d: ds[d].size, reverse=True)[0]
        else:
            return sorted(
                ds.nd.dims, key=lambda d: ds[d].size, reverse=True)[0]

    def _buffer(self, dim):
        """
        Given the dimension to parallelize, return the required buffer.
        """
        if dim not in self.dims:
            return 0
        else:
            axis = self.dims.index(dim)
            return self.r[axis] + self.f[axis]

    def _filter(self, arr, axes, output):
        #
        # Pad r and f to three dimensions.
        #
        pad_before = np.zeros(4 - arr.ndim, dtype=self.r.dtype)
        pad_after = np.zeros(arr.ndim - len(self.r) - 1, dtype=self.r.dtype)
        r = np.concatenate([pad_before, self.r, pad_after])
        f = np.concatenate([pad_before, self.f, pad_after])
        #
        # Pad input and output to four dimensions (three dimensions plus
        # variables).
        #
        values = np.array(arr, ndmin=4, copy=False)
        _out = np.array(output, ndmin=4, copy=False)

        _pixelwise_nlmeans_3d(values, _out, r, f, self.sigma, self.h,
                              self.n_eff)

    def _pixelfilter(self, pixel, output):
        ...


nlmeans = wrap_algorithm(NLMeansFilter, 'nlmeans')


# ------------
# STDEV FILTER
# ------------

# class StdevFilter(Filter):

#     def __init__(self, dims=('y', 'x'), w=3):
#         self.dims = tuple(dims)
#         self.w = w

#     def _filter(self, arr, axes, output):
#         # Generate kernel
#         kernel_shape = np.ones(arr.ndim, dtype=int)
#         kernel_shape[list(axes)] = self.w
#         kernel = np.ones(kernel_shape)

#         # Compute standard deviation over window
