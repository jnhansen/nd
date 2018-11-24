"""
This module provides read/write capabilities for satellite data.
Internally all data is handled as xarray Datasets.

TODO: For any operation that affects nrows, ncols, or extent,
need to update the metadata!

"""
from .gdal_ import from_gdal_dataset
from .netcdf_ import from_netcdf, to_netcdf
from .beamdimap_ import from_beam_dimap
from .convert_ import disassemble_complex, assemble_complex, add_time


__all__ = ['from_gdal_dataset',
           'from_beam_dimap',
           'to_netcdf',
           'from_netcdf',
           'assemble_complex',
           'disassemble_complex',
           'add_time']
