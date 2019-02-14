"""
This module contains all functionality related to reprojecting a
dataset.
"""

from .warp_ import (Reprojection, reproject,
                    Resample, resample,
                    Alignment, align,
                    _parse_crs,
                    get_crs,
                    get_transform, get_resolution, get_bounds, get_extent,
                    nrows, ncols,
                    get_common_bounds, get_common_extent,
                    get_common_resolution)

from .coregister_ import Coregistration, coregister

__all__ = ['Reprojection',
           'reproject',
           'Resample',
           'resample',
           'Alignment',
           'align',
           'Coregistration'
           'coregister'
           '_parse_crs',
           'get_crs',
           'get_transform',
           'get_resolution',
           'get_bounds',
           'get_extent',
           'nrows',
           'ncols',
           'get_common_bounds',
           'get_common_extent',
           'get_common_resolution']
