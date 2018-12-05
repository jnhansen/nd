"""
This module contains all functionality related to reprojecting a
dataset.
"""

from .warp_ import Reprojection, Alignment, _parse_crs, get_crs, \
                   get_transform, get_resolution, get_bounds, get_extent, \
                   get_common_extent

__all__ = ['Reprojection', 'Alignment', '_parse_crs', 'get_crs',
           'get_transform', 'get_resolution', 'get_bounds', 'get_extent',
           'get_common_extent']
