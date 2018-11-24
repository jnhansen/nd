from .convert_ import assemble_complex, disassemble_complex
import xarray as xr


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
    if 'encoding' not in kwargs:
        comp = dict(zlib=True, complevel=5)
        kwargs['encoding'] = {var: comp for var in write.data_vars}
    return write.to_netcdf(path, *args, **kwargs)


def from_netcdf(path, *args, **kwargs):
    """Wrapper function for `xarray.open_dataset` that preserves complex
    valued data.

    Parameters
    ----------
    path : str
        The path of the NetCDF file to read.

    Returns
    -------
    xarray.Dataset
    """
    # Make sure to load as lazy dask arrays:
    if 'chunks' not in kwargs:
        kwargs['chunks'] = {}
    ds = xr.open_dataset(path, *args, **kwargs)
    ds = assemble_complex(ds)
    return ds
