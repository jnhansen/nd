"""
The main use of image filters is for noise reduction.
This module implements several such filters, all of which are designed to work
in an arbitrary number of dimensions.

.. document private functions
.. autofunction:: _expand_kernel
"""

from .filter_ import Filter
from .nlmeans_ import NLMeansFilter, nlmeans
from .convolve_ import (ConvolutionFilter, convolution,
                        BoxcarFilter, boxcar,
                        GaussianFilter, gaussian,
                        _expand_kernel)

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
