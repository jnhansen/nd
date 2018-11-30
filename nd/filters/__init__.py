"""
The main use of image filters is for noise reduction.
This module implements several such filters, all of which are designed to work
in an arbitrary number of dimensions.

.. document private functions
.. autofunction:: _expand_kernel
"""

from .nlmeans_ import nlmeans
from .convolve_ import convolve, boxcar, gaussian, _expand_kernel

__all__ = ['convolve', 'boxcar', 'nlmeans', 'gaussian', '_expand_kernel']
