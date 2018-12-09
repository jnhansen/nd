.. _readwrite:

============================
Reading and writing datasets
============================

As ``nd`` is built around ``xarray``, most of the IO is handled by ``xarray``.
However, the :mod:`nd.io` module provides an extra layer of abstraction, specifically with the functions
:meth:`nd.io.open_dataset` and :meth:`nd.io.to_netcdf`.


Reading a dataset
-----------------
Ideally, your data exists already in the netCDF file format.
However, as most geospatial data is distributed as GeoTiff or other file formats, ``nd`` and ``xarray``
rely on `rasterio <https://rasterio.readthedocs.io/en/latest/>`_ as a Python-friendly wrapper around GDAL for dealing with such raster data.

The :mod:`nd.io` module contains three functions to read different file formats:

 * :meth:`nd.io.open_netcdf` to read a NetCDF file

 * :meth:`nd.io.open_rasterio` to read a rasterio/GDAL readable file

 * :meth:`nd.io.open_beam_dimap` to read the BEAM Dimap format, which is the best supported format in `SNAP <http://step.esa.int/main/toolboxes/snap/>`_.

as well as the convenience function :meth:`nd.io.open_dataset` which calls one of the three functions above based on the file extension.
All of these return ``xarray.Dataset`` or ``xarray.DataArray`` objects.

Most of the algorithms work on both Dataset and DataArray objects.


Writing a dataset
-----------------
Write your processed data to disk using :meth:`nd.io.to_netcdf`.

.. Note::

   Currently, it is assumed that you will only ever want to convert your data from other formats into netCDF, but not the other way around. So if you need to export your result as a GeoTiff, you are on your own (for now). Sorry about that!


Here is a list of things that ``nd.io.open_netcdf`` and ``nd.io.to_netcdf`` do in addition to ``xarray.open_dataset`` and ``xarray.Dataset.to_netcdf``:

 * Handle complex-valued data. NetCDF doesn't support complex valued data, so before writing to disk, complex variables are disassembled into their real and imaginary parts. After reading from disk, these parts are reassembled into complex valued variables. That means if you use the functions provided by ``nd.io`` you don't have to worry about complex-valued data at all.

 * Provide ``x`` and ``y`` coordinate arrays even if the NetCDF file uses ``lat`` and ``lon`` nomenclature. This is to be consistent with the general case of arbitrary projections.

::

    >>> from nd.io import open_dataset
    >>> import xarray as xr
    >>> path = 'data/C2.nc'
    >>> ds_nd = open_dataset(path)
    >>> ds_xr = xr.open_dataset(path)
    >>> {v: ds_nd[v].dtype for v in ds_nd.data_vars}
    {'C11': dtype('<f4'),
     'C22': dtype('<f4'),
     'C12': dtype('complex64')}
    >>> {v: ds_xr[v].dtype for v in ds_xr.data_vars}
    {'C11': dtype('float32'),
     'C12__im': dtype('float32'),
     'C12__re': dtype('float32'),
     'C22': dtype('float32')}


.. topic:: See Also:

 * `<http://xarray.pydata.org/en/stable/io.html>`_