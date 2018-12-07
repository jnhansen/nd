import xarray as xr
from .convert_ import assemble_complex, disassemble_complex


def to_netcdf(ds, path, *args, **kwargs):
    """Write an xarray Dataset to disk.

    In addition to xarray.to_netcdf, this function allows to store complex
    valued data by converting it to a a pair of reals. This process is
    reverted when reading the file via `from_netcdf`.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be stored to disk.
    path : str
        The path of the target NetCDF file.
    """

    write = disassemble_complex(ds)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in write.variables}
    if 'encoding' in kwargs:
        encoding.update(kwargs['encoding'])
    kwargs['encoding'] = encoding
    return write.to_netcdf(path, *args, **kwargs)


def open_netcdf(path, *args, **kwargs):
    """Read a NetCDF file into an xarray Dataset.

    Wrapper function for `xarray.open_dataset` that preserves complex
    valued data.

    Parameters
    ----------
    path : str
        The path of the NetCDF file to read.
    *args : list
        Extra positional arguments passed on to ``xarray.open_dataset``.
    **kwargs : dict
        Extra keyword arguments passed on to ``xarray.open_dataset``.

    Returns
    -------
    xarray.Dataset
        The opened dataset.

    See Also
    --------
    * ``xarray.open_dataset``
    """

    # Make sure to load as lazy dask arrays:
    if 'chunks' not in kwargs:
        kwargs['chunks'] = {}
    ds = xr.open_dataset(path, *args, **kwargs)
    ds = assemble_complex(ds)
    #
    # If the dataset dimensions are named lon and lat,
    # rename them to x and y for consistency.
    # Retain lat and lon as separate coordinates.
    #
    if 'lon' in ds.dims and 'lat' in ds.dims:
        ds = ds.rename({'lat': 'y', 'lon': 'x'})
        ds.coords['lat'] = ds.coords['y']
        ds.coords['lon'] = ds.coords['x']
    return ds
