# Strangely, h5netcdf must be imported at this point
# for the h5netcdf backend to work properly on Travis (and Ubuntu)
try:
    import h5netcdf
except Exception:
    pass

from xarray import Dataset, DataArray
from .io import open_dataset
from .visualize import to_rgb, write_video

__all__ = ['change', 'classify', 'io', 'visualize', 'filters', 'utils', 'warp',
           # Non-modules:
           'open_dataset',
           'to_rgb',
           'write_video'
           ]
