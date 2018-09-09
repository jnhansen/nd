# geotools [*work in progress*]

This package contains a selection of tools to handle and analyze satellite data.

`geotools` is making heavy use of the `xarray` library. `dask` is used for parallelization.

The GDAL library is only used as a compatibility layer in `geotools.satio` to enable reading supported file formats.
Internally, all data is passed around as `xarray` Datasets and all provided functions expect this format as inputs.
`geotools.satio.from_gdal_dataset` may be used to convert any `gdal.Dataset` object or GDAL-readable file into an `xarray` Dataset.


---

## Modules

### satio
Several functions to read and write satellite data.
* to/from NetCDF
* read data from open GDAL datasets and any GDAL-readable file
* deal with complex-valued data (not supported by NetCDF) by disassembling into two reals when writing to NetCDF, and vice versa when reading.


### change
Implements a change detection algorithm by Conradsen et al. (2015)
* convert dual polarization data into the complex covariance matrix representation
* omnibus test


### utils
Several utility functions.
* split/merge numpy arrays, xarray datasets, ...
* parallelize operations acting on xarray datasets


### warp
Given a dataset with Grounds Control Points (GCPs), usually in the form of a tie point grid,
warp the dataset onto an equirectangular projection (WGS84), such that lat/lon directly correspond to the
y and x coordinates, resepctively.

This makes concatenating datasets easier and reduces storage size, because lat/lon coordinates
do not need to be stored for each pixel.


### visualize
Several functions to quickly visualize data.

*Work in progress*


### tiling
* Split a dataset into tiles.
* Read a tiled dataset.
* Map a function across a tiled dataset.
* Create and merge tiles with buffer to avoid edge affects.
