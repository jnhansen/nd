[![Build Status](https://travis-ci.com/jnhansen/nd.svg?branch=master)](https://travis-ci.com/jnhansen/nd)
[![codecov](https://codecov.io/gh/jnhansen/nd/branch/master/graph/badge.svg)](https://codecov.io/gh/jnhansen/nd)
[![Documentation](https://readthedocs.org/projects/nd/badge/?version=latest)](https://nd.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/nd.svg)](https://badge.fury.io/py/nd)


# nd

## Overview

The main goal of this library is to generalize
methods that work in lower dimensions to higher-dimensional data.

Multi-dimensional data often arises as spatio-temporal datacubes,
e.g. climate data or time series of geospatial satellite data.
Many data analysis methods are designed to work on single images
or time series at a single point.
`nd` makes it easy to broadcast these methods across a whole dataset,
adding additional features such as automatic parallelization.

Examples include

- pixelwise change detection algorithms
- reprojection between coordinate systems
- machine learning algorithms

`nd` is built on `xarray`.
Internally, all data is passed around as ``xarray`` Datasets and all provided methods expect this format as inputs.
An ``xarray.Dataset`` is essentially a Python representation of the NetCDF file format and as such easily reads/writes NetCDF files.


<!-- This package contains a selection of tools to handle and analyze satellite data. -->
``nd`` is making heavy use of the ``xarray`` and ``rasterio`` libraries.
The GDAL library is only used via ``rasterio`` as a compatibility layer to enable reading supported file formats.
`nd.open_dataset` may be used to read any NetCDF file or any GDAL-readable file into an ``xarray.Dataset``.

Read the [Documentation](https://nd.readthedocs.io/en/latest/) for detailed user guides.

## Installation

```
pip install nd
```

It is recommended that you have GDAL available before installation and also make sure to have the correct environment variable set:

```bash
export GDAL_DATA=$(gdal-config --datadir)
```

Note that the following algorithms require the ``libgsl-dev`` C library to be installed:

- ``nd.change.OmnibusTest``


## What does this library add?

``xarray`` provides all data structures required for dealing with `n`-dimensional data in Python. ``nd`` explicitly does not aim to add additional data structures or file formats.
Rather, the aim is to bring the various corners of the scientific ecosystem in Python closer together.

As such, ``nd`` adds functionality to more seamlessly integrate libraries like ``xarray``, ``rasterio``, ``scikit-learn``, etc.

For example:

 * ``nd`` allows to reproject an entire multivariate and multi-temporal dataset between different coordinate systems by wrapping ``rasterio`` methods.

 * ``nd`` provides a wrapper for ``scikit-learn`` estimators to easily apply classification algorithms to raster data.

Additionally, ``nd`` contains a growing library of algorithms that are especially useful for spatio-temporal datacubes, for example:

 * change detection algorithms

 * spatio-temporal filters

Since ``xarray`` is our library of choice for representing geospatial raster data, this is also an attempt to promote the use of ``xarray`` and the NetCDF file format in the Earth Observation community.


## Why NetCDF?

NetCDF (specifically NetCDF-4) is a highly efficient file format that was built on top of HDF5. It is capable of random access which ties in with indexing and slicing in ``numpy``.
Because slices of a large dataset can be accessed independently, it becomes feasible to handle larger-than-memory file sizes. NetCDF-4 also supports data compression using ``zlib``. Random access capability for compressed data is maintained through data chunking.
Furthermore, NetCDF is designed to be fully self-descriptive. Crucially, it has a concept of named dimensions and coordinates, can store units and arbitrary metadata.


## Feature requests and bug reports

For feature requests and bug reports please [submit an issue](https://github.com/jnhansen/nd/issues/new/choose) on the Github repository.
