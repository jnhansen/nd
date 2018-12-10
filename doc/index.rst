nd
==

This package contains a selection of tools to handle and analyze satellite data.
``nd`` is making heavy use of the ``xarray`` and ``rasterio`` libraries.
The GDAL library is only used via ``rasterio`` as a compatibility layer in ``nd.io`` to enable reading supported file formats.
Internally, all data is passed around as ``xarray`` Datasets and all provided methods expect this format as inputs.
:meth:`nd.io.open_rasterio` may be used to convert any GDAL-readable file into an ``xarray.Dataset``.

An ``xarray.Dataset`` is essentially a Python representation of the NetCDF file format and as such easily reads/writes NetCDF files.


What does this library add?
---------------------------
``xarray`` provides all data structures required for dealing with `n`-dimensional data in Python. ``nd`` explicitly does not aim to add additional data structures or file formats.
Rather, the aim is to bring the various corners of the scientific ecosystem in Python closer together.

As such, ``nd`` adds functionality to more seamlessly integrate libraries like ``xarray``, ``rasterio``, ``scikit-learn``, etc.

For example:

 * ``nd`` allows to reproject an entire multivariate and multi-temporal dataset between different coordinate systems by wrapping ``rasterio`` methods.

 * ``nd`` provides a wrapper for scikit-learn estimators to easily apply classification algorithms to raster data [in progress].

Additionally, ``nd`` contains a growing library of algorithms that are especially useful for spatio-temporal datacubes, for example:

 * change detection algorithms

 * spatio-temporal filters

Since ``xarray`` is our library of choice for representing geospatial raster data, this is also an attempt to promote the use of ``xarray`` and the NetCDF file format in the Earth Observation community.


Why NetCDF?
-----------
NetCDF (specifically NetCDF-4) is a highly efficient file format that was built on top of HDF5. It is capable of random access which ties in with indexing and slicing in ``numpy``.
Because slices of a large dataset can be accessed independently, it becomes feasible to handle larger-than-memory file sizes. NetCDF-4 also supports data compression using ``zlib``. Random access capability for compressed data is maintained through data chunking.
Furthermore, NetCDF is designed to be fully self-descriptive. Crucially, it has a concept of named dimensions and coordinates, can store units and arbitrary metadata.


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Content

   user_guide
   reference
