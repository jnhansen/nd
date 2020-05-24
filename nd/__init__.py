import os
# Strangely, h5netcdf must be imported at this point
# for the h5netcdf backend to work properly on Travis (and Ubuntu)
try:
    import h5netcdf
except Exception:
    pass

# Skip on mock install
if os.environ.get('READTHEDOCS') != 'True':
    # Import shapely.geometry at this point to avoid the error:
    # Assertion failed: (0), function query, file AbstractSTRtree.cpp
    # which may occur if shapely is imported *after* fiona
    import shapely.geometry
    import shapely.affinity

from xarray import Dataset, DataArray
from .io import open_dataset, to_netcdf
from .visualize import to_rgb, write_video
from .algorithm import Algorithm
from .tiling import auto_merge

# Import the following modules by default.
# Modules that need to be imported individually:
#  - nd.classify
from . import (change, io, visualize, filters, utils, warp, vector)
from . import _xarray

__all__ = ['change', 'io', 'visualize', 'filters', 'utils', 'warp',
           'vector',
           # Non-modules:
           'Algorithm',
           'open_dataset',
           'to_netcdf',
           'to_rgb',
           'write_video',
           'auto_merge'
           ]
