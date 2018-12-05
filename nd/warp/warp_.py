import glob
import os
import numpy as np
import xarray as xr
import rasterio.warp
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.errors import CRSError
from affine import Affine
from ..algorithm import Algorithm
from ..io import from_netcdf, to_netcdf


def _parse_crs(crs):
    """
    Parse a coordinate reference system from a variety of representations.

    Parameters
    ----------
    crs : {str, dict, int, CRS}
        Must be either a rasterio CRS object, a proj-string, rasterio supported
        dictionary, WKT string, or EPSG integer.

    Returns
    -------
    rasterio.crs.CRS
        The parsed CRS.

    Raises
    ------
    CRSError
        Raises an error if the input cannot be parsed.
    """

    if isinstance(crs, CRS):
        parsed = crs
    elif isinstance(crs, str):
        try:
            # proj-string
            parsed = CRS.from_string(crs)
        except CRSError:
            # wkt
            parsed = CRS.from_wkt(crs)
    elif isinstance(crs, dict):
        parsed = CRS(crs)
    elif isinstance(crs, int):
        parsed = CRS.from_epsg(crs)
    else:
        raise CRSError('Could not parse CRS.')

    return parsed


def get_crs(ds, format='crs'):
    """
    Extract the Coordinate Reference System from a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset
    format : str {'crs', 'proj', 'dict', 'wkt'}
        The format in which to return the CRS.

        * 'proj': A proj-string, e.g. ``+init=epsg:4326``

        * 'dict': e.g. ``{'init': 'EPSG:4326'}``

        * 'wkt': e.g. ``GEOGCS["WGS 84", ...]``

    Returns
    -------
    CRS, str, or dict
        The CRS.
    """

    if 'crs' not in ds.attrs:
        return None

    crs = _parse_crs(ds.attrs['crs'])

    if format == 'crs':
        return crs
    if format == 'proj':
        return crs.to_string()
    if format == 'dict':
        return crs.to_dict()
    if format == 'wkt':
        return crs.wkt


def get_transform(ds):
    """
    Extract the geographic transform from a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset

    Returns
    -------
    affine.Affine
        The affine transform
    """

    if 'transform' not in ds.attrs:
        return None

    ds_trans = ds.attrs['transform']
    if isinstance(ds_trans, Affine):
        return ds_trans
    else:
        return Affine(*ds_trans)


def get_resolution(ds):
    """
    Extract the resolution of the dataset in projection coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset

    Returns
    -------
    tuple
        The raster resolution as (x, y)
    """

    if 'res' in ds.attrs:
        return ds.attrs['res']

    return None


def get_bounds(ds):
    """
    Extract the bounding box in projection coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset

    Returns
    -------
    tuple
        The bounding box in projection coordinates
        (left, bottom, right, top).
    """

    trans = get_transform(ds)
    if trans is not None:
        nrows = ds.dims['y']
        ncols = ds.dims['x']
        corners = (np.array([0, 0, ncols-1, ncols-1]),
                   np.array([0, nrows-1, 0, nrows-1]))
        corner_x, corner_y = trans * corners
        return BoundingBox(
            left=corner_x.min(),
            bottom=corner_y.min(),
            right=corner_x.max(),
            top=corner_y.max()
        )
    else:
        return BoundingBox(
            left=ds['x'].min(),
            bottom=ds['y'].min(),
            right=ds['x'].max(),
            top=ds['y'].max()
        )


def get_extent(ds):
    """
    Extract the extent (bounding box) from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset

    Returns
    -------
    tuple
        The extent (left, bottom, right, top) in latitude and longitude
        coordinates.
    """

    #
    # Check if latitude and longitude are stored as coordinates.
    #
    if 'lon' in ds.coords and 'lat' in ds.coords:
        return BoundingBox(
            left=ds.lon.values.min(),
            bottom=ds.lat.values.min(),
            right=ds.lon.values.max(),
            top=ds.lat.values.max()
        )

    #
    # Otherwise, get extent from projection information
    # by projecting the corner coordinates onto EPSG:4326
    # to obtain the latitude and longitude at the four corners.
    #
    src_crs = get_crs(ds)
    if src_crs is None:
        raise CRSError('Could not determine the CRS.')

    dst_crs = CRS(init='epsg:4326')
    proj_bounds = get_bounds(ds)
    bounds = rasterio.warp.transform_bounds(
        src_crs, dst_crs, **proj_bounds._asdict()
    )
    return BoundingBox(*bounds)


def get_common_extent(datasets):
    """
    Calculate the smallest extent that contains all of the input datasets.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        The input datasets.

    Returns
    -------
    tuple
        The common extent (left, bottom, right, top) in latitude and longitude
        coordinates.
    """

    extents = []
    for ds in datasets:
        extent = get_extent(ds)
        extents.append(extent)
        # lon_min, lat_min, lon_max, lat_max = extent

    # Get largest extent:
    extents = np.array(extents)
    common_extent = np.concatenate((extents[:, :2].min(axis=0),
                                    extents[:, 2:].max(axis=0)))

    return common_extent


class Reprojection(Algorithm):
    """
    Reprojection of the dataset to the given coordinate reference system
    (CRS) and extent.

    Parameters
    ----------
    crs : dict or str
        The output coordinate reference system as dictionary or proj-string
    extent : tuple, optional
        The output extent. By default this is determined from the input data.
    """

    def __init__(self, crs, extent=None, res=None, width=None, height=None,
                 transform=None):
        if transform is not None and (width is None or height is None):
            raise ValueError('If `transform` is given, you must also specify '
                             'the `width` and `height` arguments.')

        self.crs = _parse_crs(crs)
        self.extent = extent
        self.res = res
        self.width = width
        self.height = height
        self.transform = transform

    def apply(self, ds):
        """
        Apply the projection to a dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset.

        Returns
        -------
        xarray.Dataset
            The reprojected dataset.
        """
        src_crs = get_crs(ds)
        src_transform = get_transform(ds)
        src_bounds = get_bounds(ds)

        if self.transform is None:
            transform, width, height = \
                rasterio.warp.calculate_default_transform(
                    src_crs, self.crs,
                    ds.dims['x'], ds.dims['y'],
                    resolution=self.res,
                    dst_width=self.width,
                    dst_height=self.height,
                    **src_bounds._asdict())
        else:
            transform = self.transform
            width = self.width
            height = self.height

        dst_shape = (height, width)
        dst_dims = ('y', 'x')

        dst_x, _ = transform * (np.arange(width), np.zeros(width, dtype=int))
        _, dst_y = transform * (np.zeros(height, dtype=int), np.arange(height))
        dst_coords = {'x': dst_x, 'y': dst_y}

        # result = ds.copy(deep=False)
        result = xr.Dataset(coords=dst_coords)
        result.attrs = ds.attrs

        for v in ds.data_vars:
            output = np.zeros(dst_shape, dtype=ds[v].dtype)
            output[:] = np.nan
            rasterio.warp.reproject(
                ds[v].values,
                output,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=self.crs,
                resampling=rasterio.warp.Resampling.bilinear
            )
            result[v] = (dst_dims, output)

        # Serialize transform to tuple and store in metadata
        result.attrs['transform'] = transform[:6]
        # Store CRS info in metadata
        result.attrs['crs'] = self.crs.to_string()
        result.attrs['coordinate_system_string'] = self.crs.wkt
        # Store new data shape in metadata
        result.attrs['lines'] = result.dims['y']
        result.attrs['samples'] = result.dims['x']

        return result


class Alignment(Algorithm):
    """
    Align a list of datasets to the same coordinate grid.

    Parameters
    ----------
    crs : str or dict, optional
        The coordinate reference system as proj-string or dictionary.
        By default, use the CRS of the datasets.
    extent : tuple, optional
        The bounding box of the output dataset. By default, use the common
        extent of all datasets.
    """

    def __init__(self, crs=None, extent=None):
        self.crs = crs
        self.extent = extent

    def apply(self, datasets, path):
        """
        Resample datasets to common extent and resolution.

        Parameters
        ----------
        datasets : str, list of str, list of xarray.Dataset
            The input datasets. Can be either a glob expression,
            a list of filenames, or a list of opened datasets.
        path : str
            The output path to store the aligned datasets.
        """

        if self.extent is None:
            extent = get_common_extent(datasets)
        else:
            extent = self.extent

        # Treat `datasets` as a glob expression
        if isinstance(datasets, str):
            datasets = glob.glob(datasets)

        if len(datasets) == 0:
            raise ValueError("No files found!")

        # Treat `datasets` as a list of file paths
        products = datasets
        if isinstance(datasets[0], str):
            # Pass chunks={} to ensure the dataset is read as a dask array
            product_names = [os.path.splitext(os.path.split(_)[1])[0]
                             for _ in products]
            datasets = [from_netcdf(path) for path in datasets]
        else:
            product_names = [ds.metadata.attrs['Abstracted_Metadata:PRODUCT']
                             if 'metadata' in ds else 'data{}'.format(i)
                             for i, ds in enumerate(datasets)]

        os.makedirs(path, exist_ok=True)

        proj = Reprojection(crs=self.crs, extent=extent)
        for name, ds in zip(product_names, products):
            outfile = os.path.join(path, name + '_aligned.nc')
            if isinstance(ds, str):
                ds = from_netcdf(ds)
            res = proj.apply(ds)
            to_netcdf(res, outfile)
