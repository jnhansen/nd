geo
===

This package contains a selection of tools to handle and analyze satellite data.

`geo` is making heavy use of the `xarray` library. `dask` is used for parallelization.

The GDAL library is only used as a compatibility layer in `geo.io` to enable reading supported file formats.
Internally, all data is passed around as `xarray` Datasets and all provided functions expect this format as inputs.
`geo.io.from_gdal_dataset` may be used to convert any `gdal.Dataset` object or GDAL-readable file into an `xarray` Dataset.


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

   overview
   user_guide
   reference



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
