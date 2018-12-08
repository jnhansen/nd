"""
This module contains all functionality related to reprojecting a
dataset.
"""

from .warp_ import Reprojection, Resample, Alignment, _parse_crs, get_crs, \
                   get_transform, get_resolution, get_bounds, get_extent, \
                   get_common_bounds, get_common_extent, get_common_resolution

__all__ = ['Reprojection',
           'Resample',
           'Alignment',
           '_parse_crs',
           'get_crs',
           'get_transform',
           'get_resolution',
           'get_bounds',
           'get_extent',
           'get_common_bounds',
           'get_common_extent',
           'get_common_resolution']
