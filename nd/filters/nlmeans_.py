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

    def __init__(self, dims, r, sigma, h, f=1):
        if isinstance(r, (int, float)):
            r = [r] * len(dims)
        self.dims = tuple(dims)
        # Pad r and f to three dimensions
        _r = np.array(r, dtype=np.uint32)
        _f = np.array([f if _ > 0 else 0 for _ in _r], dtype=np.uint32)
        pad = np.zeros(3 - len(_r), dtype=_r.dtype)
        self.r = np.concatenate([pad, _r])
        self.f = np.concatenate([pad, _f])
        self.sigma = sigma
        self.h = h

    def _filter(self, arr, axes, output):
        values = np.array(arr, ndmin=4, copy=False)
        _out = np.array(output, ndmin=4, copy=False)
        _pixelwise_nlmeans_3d(values, _out, self.r, self.f,
                              self.sigma, self.h)
