"""
This module provides read/write capabilities for satellite data.
Internally all data is handled as xarray Datasets.

TODO: For any operation that affects nrows, ncols, or extent,
need to update the metadata!

"""
from .open_ import open_dataset
from .netcdf_ import open_netcdf, to_netcdf
from .beamdimap_ import open_beam_dimap
from .rasterio_ import open_rasterio
from .convert_ import disassemble_complex, assemble_complex, add_time


__all__ = ['open_dataset',
           'open_netcdf',
           'open_beam_dimap',
           'open_rasterio',
           'to_netcdf',
           'assemble_complex',
           'disassemble_complex',
           'add_time']
