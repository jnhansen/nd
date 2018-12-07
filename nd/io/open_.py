import os
from .beamdimap_ import open_beam_dimap
from .netcdf_ import open_netcdf
from .rasterio_ import open_rasterio


def open_dataset(path, *args, **kwargs):
    """
    The default way of reading a dataset from disk.

    Determines the file format from the extension, and calls either
    :meth:`nd.io.open_netcdf`, :meth:`nd.io.open_beam_dimap`, or
    :meth:`nd.io.open_rasterio`.

    Parameters
    ----------
    path : str
        The file path.
    *args : list
        Extra positional arguments passed on to the specialized ``open_*``
        function.
    **kwargs : dict
        Extra keyword arguments passed on to the specialized ``open_*``
        function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The opened dataset. In general, if the file is a NetCDF or BEAM-Dimap
        file the result will be an xarray Dataset, otherwise an xarray
        DataArray.

    Raises
    ------
    IOError
        Raises an IOError if the Dataset could not be opened.
    """

    _, ext = os.path.splitext(path)
    if ext == '.nc':
        return open_netcdf(path, *args, **kwargs)

    elif ext == '.dim':
        return open_beam_dimap(path, *args, **kwargs)

    try:
        return open_rasterio(path, *args, **kwargs)
    except Exception:
        raise IOError('Could not read the file.')
