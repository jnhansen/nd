import numpy as np
from ._nlmeans import _pixelwise_nlmeans_3d
from .filter_ import Filter


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
    """

    per_variable = False

    def __init__(self, dims, r=1, sigma=1, h=1, f=1):
        if isinstance(r, (int, float)):
            r = [r] * len(dims)
        self.dims = tuple(dims)
        self.r = np.array(r, dtype=np.uint32)
        self.f = np.array([f if _ > 0 else 0 for _ in self.r], dtype=np.uint32)
        self.sigma = sigma
        self.h = h

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

        _pixelwise_nlmeans_3d(values, _out, r, f, self.sigma, self.h)

    def _pixelfilter(self, pixel, output):
        ...
