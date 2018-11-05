[![Build Status](https://travis-ci.com/jnhansen/geo.svg?branch=master)](https://travis-ci.com/jnhansen/geo)


# geo [*work in progress*]

This package contains a selection of tools to handle and analyze satellite data.

`geo` is making heavy use of the `xarray` library. `dask` is used for parallelization.

The GDAL library is only used as a compatibility layer in `geo.io` to enable reading supported file formats.
Internally, all data is passed around as `xarray` Datasets and all provided functions expect this format as inputs.
`geo.io.from_gdal_dataset` may be used to convert any `gdal.Dataset` object or GDAL-readable file into an `xarray` Dataset.


---

## Submodules

### geo.io
Several functions to read and write satellite data.
* to/from NetCDF
* read data from open GDAL datasets and any GDAL-readable file
* deal with complex-valued data (not supported by NetCDF) by disassembling into two reals when writing to NetCDF, and vice versa when reading.


### geo.change
A module implementing change detection algorithms.
* convert dual polarization data into the complex covariance matrix representation
* OmnibusTest (change detection algorithm by Conradsen et al. (2015))


### geo.classify
A collection of classification and clustering methods.

... *work in progress* ...


### geo.filter
Implements several filters, currently:
* kernel convolutions
* non-local means


### geo.utils
Several utility functions.
* split/merge numpy arrays, xarray datasets, ...
* parallelize operations acting on xarray datasets


### geo.warp
Given a dataset with Ground Control Points (GCPs), usually in the form of a tie point grid,
warp the dataset onto an equirectangular projection (WGS84), such that lat/lon directly correspond to the
y and x coordinates, resepctively.

This makes concatenating datasets easier and reduces storage size, because lat/lon coordinates
do not need to be stored for each pixel.


### geo.visualize
Several functions to quickly visualize data.
* create RGB images from data
* create video from a spatiotemporal dataset


### geo.tiling
* Split a dataset into tiles.
* Read a tiled dataset.
* Map a function across a tiled dataset.
* Create and merge tiles with buffer to avoid edge affects.
