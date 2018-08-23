"""
This module contains all functionality related to reprojecting a
dataset.

TODO: Make all functions work with xarray Datasets.
TODO: remove clutter in main()

"""
# somehow need to import gdal first ...
from geo import satio
from osgeo import gdal, osr
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
import os
import glob
from scipy.ndimage import interpolation
from scipy.ndimage._ni_support import _extend_mode_to_code
# import (_extend_mode_to_code, map_coordinates , geometric_transform)
from scipy.ndimage import _nd_image
from geo import utils
from geo._warp import c_grid, CoordTransform
import time

try:
    type(profile)
except NameError:
    def profile(fn): return fn


__all__ = ['warp', 'warp_like', 'align', 'gdal_warp', 'resample']


NO_DATA_VALUE = np.nan

# def _get_gcp_extent(src):
#     gcp_df = get_gcp_df(src)
#     lon = gcp_df['GCPX']
#     lat = gcp_df['GCPY']
#     return [lon.min(), lat.min(), lon.max(), lat.max()]


def _get_extent(ds):
    if 'lon' not in ds.coords or 'lat' not in ds.coords:
        raise ValueError("Dataset must contain 'lat' and 'lon' coordinates.")
    # [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat]
    return (ds.lon.values.min(), ds.lat.values.min(),
            ds.lon.values.max(), ds.lat.values.max())


def _tie_points_to_gcps(ds):
    """Given an input dataset with tie point grids, generate a DataFrame
    of the GCPs.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing lat/lon tie point grids.

    Returns
    -------
    pandas DataFrame
        The GCPs as DataFrame.
    """
    if 'lat' not in ds or 'lon' not in ds:
        raise ValueError("The dataset must contain 'lat' and 'lon'.")
    mask = ~np.isnan(ds.lat)
    ncols, nrows = ds.lat.shape
    y, x = np.meshgrid(np.arange(nrows), np.arange(ncols), copy=False)
    gcp_dict = {'GCPX': ds.lon.values[mask],
                'GCPY': ds.lat.values[mask],
                'GCPPixel': y[mask],
                'GCPLine': x[mask]}
    gcps = pd.DataFrame(gcp_dict)
    return gcps


@profile
def _fit_latlon(coords, degree=3, inverse=False, return_coef=False):
    """Fit a polynomial to the input coordinates.

    TODO: Make work with shape (2, M, N)
    NOTE: This function will not need to be called in the public API.

    Parameters
    ----------
    coords : numpy.array (M, N, 2) or pandas.DataFrame
        Either a numpy array of lat/lon coordinates, or a pandas DataFrame
        with the columns ['GCPLine', 'GCPPixel', 'GCPX', 'GCPY'] containing
        the Ground Control Points.
    degree : int, optional
        The polynomial degree to be fitted (default: 3)
    inverse : bool, optional
        If True, fit x,y as function of lon,lat (default: False).
    return_coef : bool, optional
        If True, return the coefficients of the fitted polynomial.
        Otherwise, return a function that converts between (x,y) and (lat,lon).
        (default: False)

    Returns
    -------
    function
        If `inverse` is False, the returned function converts (y,x) to
        (lon,lat). Otherwise, the function returns (lon,lat) for an input of
        (y,x).
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    poly = PolynomialFeatures(degree=degree)

    if isinstance(coords, pd.DataFrame):
        #
        # GCPs
        #
        must_contain = ['GCPLine', 'GCPPixel', 'GCPX', 'GCPY']
        if not coords.columns.isin(must_contain).all():
            raise ValueError("The DataFrame `coords` must contain the columns"
                             " ['GCPLine', 'GCPPixel', 'GCPX', 'GCPY'].")
        ll = coords[['GCPX', 'GCPY']]
        xy = coords[['GCPLine', 'GCPPixel']]
    elif not isinstance(coords, np.ndarray):
        raise ValueError("`coords` is not a valid numpy array.")
    elif coords.ndim != 3 or coords.shape[2] != 2:
        raise ValueError("`coords` must have shape (rows, cols, 2)."
                         " Found shape %s instead." % repr(coords.shape))
    else:
        #
        # Lat-Lon grid
        #
        y, x = np.meshgrid(np.arange(coords.shape[1]),
                           np.arange(coords.shape[0]),
                           copy=False)

        # subsample
        step = 50
        ll = coords[::step, ::step, :]
        xy = np.stack((y[::step, ::step], x[::step, ::step]), axis=-1)
        # remove nan coordinates and flatten
        mask = np.isnan(ll).any(axis=2)
        ll = ll[~mask]
        xy = xy[~mask]

    if inverse:
        regressor = poly.fit_transform(ll)
        regressand = xy
    else:
        regressor = poly.fit_transform(xy)
        regressand = ll

    clf = linear_model.LinearRegression()
    clf.fit(regressor, regressand)

    if return_coef:
        return clf.coef_

    else:
        def fn(X):
            """
            This function maps from (x,y) to (lon,lat) (or the reverse).
            """
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            orig_shape = X.shape

            if len(orig_shape) == 1:
                # single coordinate pair was passed
                X = np.array([X])
            elif len(orig_shape) == 3:
                # flatten X temporarily
                X = X.reshape(-1, 2)

            #
            # If X is very large, split into chunks.
            #
            # Empirical optimal chunk sizes (approximately):
            # chunk size | len(X) | degree
            # -----------|--------|--------
            # 8000       | 1e8    | 3
            # 8000       | 1e8    | 2
            res = np.empty_like(X)
            for index, chunk in utils.array_chunks(X, 8000, axis=0,
                                             return_indices=True):
                p = poly.transform(chunk)
                res[index] = clf.predict(p)

            # reshape back to original shape:
            if len(orig_shape) != 2:
                res = res.reshape(orig_shape)

            return res

        return fn


@profile
def _coord_transform(coords, new_coords, cdim=2):
    """Generate an array of image space coordinates that will transform
    from `coords` to `new_coords`.

    TODO: Make work with shape (2, M, N)

    Parameters
    ----------
    coords : numpy.array, shape (M, N, 2)
        The original coordinates of each pixel.
    new_coords : numpy.array, shape (Y, X, 2)
        The new coordinates.

    Returns
    -------
    numpy.array, shape (Y, X, 2)
        The image coordinates corresponding to the transform.
    """
    # 1) find the function coord2xy for coords.
    coord2xy = _fit_latlon(coords, inverse=True)
    # 2) apply coord2xy to new_coords.
    im_coords = coord2xy(new_coords)

    return im_coords


def _common_extent_and_resolution(datasets):
    extents = []
    resolutions = []
    for ds in datasets:
        lon_min = ds.lon.min()
        lat_min = ds.lat.min()
        lon_max = ds.lon.max()
        lat_max = ds.lat.max()
        extents.append([lon_min, lat_min, lon_max, lat_max])
        # can only calculate resolution if `lat` and `lon` are
        # dimensions, i.e. one-dimensional arrays
        if ds.lat.ndim == 1:
            res_lon = (lon_max - lon_min) / ds.lon.shape[0]
            res_lat = (lat_max - lat_min) / ds.lat.shape[0]
            resolutions.append([res_lon, res_lat])

    # Get largest extent:
    extents = np.array(extents)
    common_extent = np.concatenate((extents[:, :2].min(axis=0),
                                    extents[:, 2:].max(axis=0)))

    # Get best resolution:
    if len(resolutions) == 0:
        common_resolution = None
    else:
        resolutions = np.array(resolutions)
        common_resolution = resolutions.min(axis=0)

    return common_extent, common_resolution


# NOTE: deprecate?
def gdal_warp(src):
    """ Warps a GDAL dataset onto EPSG:4326, i.e. a lat-lon grid.
    https://gis.stackexchange.com/a/140053

    Parameters
    ----------
    src : osgeo.gdal.Dataset

    Returns
    -------
    osgeo.gdal.Dataset
        A dataset warped onto EPSG:4326 (lat-lon grid)
    """
    # Define target SRS
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    dst_wkt = dst_srs.ExportToWkt()

    error_threshold = 0.125
    resampling = gdal.GRA_Bilinear
    # resampling = gdal.GRA_NearestNeighbour

    # The second argument is src_wkt :
    # left to default value --> will use the one from source
    warped_ds = gdal.AutoCreateWarpedVRT(src,
                                         None,
                                         dst_wkt,
                                         resampling,
                                         error_threshold)

    return warped_ds


@profile
def resample(input, coords=None, mapping=None, output=None, order=3,
             mode='constant', cval=np.nan, copy=True, **kwargs):
    """
    This is effectively an extension of `map_coordinates` and
    `geometric_transform` in scipy.ndimage.interpolation that can handle np.nan
    values as well as complex data in the input array.

    NOTE: do spline filtering?

    Parameters
    ----------
    input : ndarray (M, N)
    coords : ndarray (2, Y, X), optional
        An array of coordinates (in input coordinate space) at which the input
        data will be resampled (default: None).
    mapping : function, optional
        A function that maps from the output coordinate space to the input
        coordinate space. This is a more memory-efficient alternative to
        `coords` (default: None).
    output : ndarray (Y, X), optional
        An array that will be modified inplace to contain the resampled input.
        If None, return a new array (default: None).
    order : int, optional
        Order of spline interpolation (default: 3).
    mode : str, optional
        How to fill out-of-bounds pixels (default: 'constant')
    cval : float, optional
        Fill value if mode is 'constant' (default: np.nan).
    copy : bool, optional
        If True, create copy of input data (default).
        Otherwise, fill NaN values inplace (more memory efficient).

    Returns
    -------
    ndarray or None
        If `output` is None, returns the resampled array. Otherwise, returns
        None.
    """
    extra_arguments = None if mapping is None else ()
    extra_keywords = None if mapping is None else {}
    mode = _extend_mode_to_code(mode)
    const_mode = _extend_mode_to_code('constant')
    return_output = (output is None)

    #
    # Prepare output array, which will be filled inplace.
    #
    if output is None:
        if coords is not None:
            shape = coords.shape
        else:
            shape = input.shape
        output = np.zeros(shape, dtype=input.dtype)
    else:
        shape = output.shape

    if np.iscomplexobj(output):
        out_dtype = np.float32 if output.dtype.itemsize == 8 \
                    else np.float64
        output = output[:, :, np.newaxis].view(out_dtype)

    #
    # Deal with NaN values
    #
    if np.issubdtype(input.dtype, np.floating):
        nanmask = np.isnan(input)
    else:
        nanmask = None

    if nanmask is not None and nanmask.any():
        # Warp a boolean array of NaN values, in order to later mask out the
        # result.
        nanmask_mapped = np.zeros(shape, dtype=bool)
        nan_order = 0
        _nd_image.geometric_transform(nanmask.astype(np.float32),
                                      mapping, coords, None, None,
                                      nanmask_mapped, nan_order, const_mode, 1,
                                      extra_arguments, extra_keywords)
        if copy:
            data = input.copy()
        else:
            data = input
        data[nanmask] = 0
    else:
        nanmask_mapped = None
        data = input

    #
    # Deal with complex data
    #
    if np.iscomplexobj(data):
        # Fill output[:, :, 0] with the real part,
        # and output[:, :, 1] with the imaginary part
        floatview = data[:, :, np.newaxis].view(out_dtype)
        for i in range(2):
            _nd_image.geometric_transform(floatview[:, :, i],
                                          mapping, coords, None, None,
                                          output[:, :, i], order, mode, cval,
                                          extra_arguments, extra_keywords)
        output = output.view(input.dtype)[:, :, 0]
    else:
        _nd_image.geometric_transform(data, mapping, coords, None, None,
                                      output, order, mode, cval,
                                      extra_arguments, extra_keywords)

    # Reapply NaN values
    if nanmask_mapped is not None:
        output[nanmask_mapped] = np.nan

    if return_output:
        return output
    else:
        return None


@profile
def warp(ds, extent=None, shape=None, resolution=None, chunks=None,
         precompute_coords=True):
    """Warp a dataset onto equirectangular coordinates. The resulting dataset
    will contain 'lat' and 'lon' as 1-dimensional coordinate variables, i.e.
    dimensions.

    TODO: need to make this consume less memory (!)
    TODO: parallelize
    TODO: make work with already orthorectified datasets √

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        A dataset that must contain coordinate variables 'lat' and 'lon',
        which may be NaN in most places (tie point grid).
    extent : list, optional
        The desired extent of the warped dataset in the form
        [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat]. If None (default),
        take the maximum extent spanned my the original dataset.
    shape : tuple, optional
        The desired shape of the output dataset. This will determine the
        resolution. Ignored if `resolution` is also passed.
    resolution : tuple, optional
        The resolution of the output dataset. If both the shape and resolution
        are `None`, the output shape will be equal to the input shape,
        which may not be desirable.
    precompute_coords : bool, optional
        Faster, but uses more memory (default: True).

    Returns
    -------
    xarray.Dataset
        A warped dataset of shape `shape`.
    """
    if not isinstance(ds, xr.Dataset) and not isinstance(ds, xr.DataArray):
        raise ValueError("`ds` must be a valid xarray Dataset or DataArray "
                         "(got: {}).".format(type(ds)))

    #
    # Use lat/lon coordinate arrays
    #
    if 'lat' in ds.dims and 'lon' in ds.dims:
        print(f'Computing new image coordinates ...'); t = time.time()
        if extent is None:
            # [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat]
            extent = _get_extent(ds)
        else:
            extent = tuple(extent)
        lon_min, lat_min, lon_max, lat_max = extent
        if resolution is not None:
            # Compute shape (a passed shape will be ignored).
            lon_res, lat_res = resolution
            lon_size = int((lon_max - lon_min) / lon_res)
            lat_size = int((lat_max - lat_min) / lat_res)
            shape = (lat_size, lon_size)
        elif shape is not None:
            lat_size, lon_size = shape
        else:
            lat_size = ds.sizes['lat']
            lon_size = ds.sizes['lon']
            shape = (lat_size, lon_size)

        # New coordinates (needed to generate the dataset)
        new_lats = np.linspace(lat_max, lat_min, lat_size)
        new_lons = np.linspace(lon_min, lon_max, lon_size)

        # Get original shape and extent
        o_lat_size = ds.sizes['lat']
        o_lon_size = ds.sizes['lon']
        o_shape = (o_lat_size, o_lon_size)
        o_extent = _get_extent(ds)

        if precompute_coords:
            # Generate coordinate grid
            coords = c_grid(o_extent, o_shape, extent, shape)
            mapping = None
        else:
            # Generate coordinate transform
            coords = None
            mapping = CoordTransform(o_extent, o_shape, extent, shape).apply

        print('--- {:.1f}s'.format(time.time() - t))

    #
    # USE GCPs
    #
    # TODO: Make work with new mechanism
    elif 'lat' in ds.coords and 'lon' in ds.coords:
        gcps = _tie_points_to_gcps(ds)
        ll2xy = _fit_latlon(gcps, inverse=True)
        if extent is None:
            # [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat]
            extent = [gcps['GCPX'].min(), gcps['GCPY'].min(),
                      gcps['GCPX'].max(), gcps['GCPY'].max()]
        if shape is None:
            shape = ds.lat.shape
        lat_size, lon_size = shape
        new_lons = np.linspace(extent[0], extent[2], lon_size)
        new_lats = np.linspace(extent[1], extent[3], lat_size)
        new_ll_coords = np.stack(np.meshgrid(new_lons, new_lats, copy=False),
                                 axis=-1)
        coords = ll2xy(new_ll_coords)

    else:
        raise ValueError("Could not determine the lat and lon coordinates "
                         "for the dataset.")

    #
    # Create new dataset
    #
    ds_coords = {'lat': new_lats, 'lon': new_lons}
    if 'time' in ds.coords:
        ds_coords['time'] = ds.time

    # Warp a single DataArray
    def _warp_dataarray(da):
        var = da.name
        if 'lat' not in da.coords or \
                'lon' not in da.coords:
            return da
        else:
            print(f'Warping {var} ...'); t = time.time()
            kwargs = {}
            # If the dtype is integer, treat as categorical
            # TODO: Make this optional?
            if np.issubdtype(da.dtype, np.integer):
                kwargs['order'] = 0
            new_data = np.zeros(shape, dtype=da.dtype)
            resample(da.values, coords=coords, mapping=mapping,
                     output=new_data, copy=False, **kwargs)
            da_warped = xr.DataArray(new_data, dims=ds_coords.keys(),
                                     coords=ds_coords, attrs=da.attrs)
            print('--- {:.1f}s'.format(time.time() - t))
            return da_warped

    # If requested, split into small chunks for memory reasons.
    if chunks is not None:
        parts = utils.xr_split(ds, dim='lat', chunks=chunks)
    else:
        parts = [ds]

    result = None
    for part in parts:
        if isinstance(part, xr.Dataset):
            if result is None:
                # First part only
                result = xr.Dataset(coords=ds_coords, attrs=ds.attrs)
            for var in part.data_vars:
                warped = _warp_dataarray(part[var])
                if var not in result:
                    result[var] = warped
                else:
                    result[var] = result[var].fillna(warped)

        elif isinstance(part, xr.DataArray):
            warped = _warp_dataarray(part)
            if result is None:
                # First part only
                result = warped
            else:
                result = result.fillna(warped)

    return result


def warp_like(ds, template, precompute_coords=True):
    """
    Warp `ds` to match the coordinates of `template`.
    """
    extent = _get_extent(template)
    shape = (template.sizes['lat'],
             template.sizes['lon'])
    return warp(ds, extent=extent, shape=shape,
                precompute_coords=precompute_coords)


@profile
def align(datasets, path, parallel=False, compute=True,
          precompute_coords=True):
    """
    Resample datasets to common extent and resolution.
    """
    #
    # Doing this in parallel is apparently too heavy on memory...
    #
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
        datasets = [satio.from_netcdf(path) for path in datasets]
    else:
        product_names = [ds.metadata.attrs['Abstracted_Metadata:PRODUCT']
                         for ds in datasets]

    extent, resolution = _common_extent_and_resolution(datasets)
    print(extent, resolution)

    os.makedirs(path, exist_ok=True)

    def _align(ds, outfile):
        if isinstance(ds, str):
            ds = satio.from_netcdf(ds)
        resampled = warp(ds, extent=extent, resolution=resolution,
                         precompute_coords=precompute_coords)
        satio.to_netcdf(resampled, outfile)
        # Explicitly close datasets
        del resampled
        ds.close()

    tasks = []
    for name, ds in zip(product_names, products):
        outfile = os.path.join(path, name + '_aligned.nc')
        if parallel:
            tasks.append(
                delayed(_align)(ds, outfile)
            )
        else:
            print('Aligning {}'.format(name))
            _align(ds, outfile)

    if parallel:
        result = delayed(tasks)
        if compute:
            return result.compute()
        else:
            return result


@profile
def main():
    path = '/Users/jhansen/data/alignment_test/'
    # align(path+'*.nc', path+'aligned/', parallel=False)

    # Just do one.
    datasets = glob.glob(path+'*.nc')
    datasets = [satio.from_netcdf(p) for p in datasets]
    extent, resolution = _common_extent_and_resolution(datasets)
    ds = datasets[0]
    # del ds['C11']
    # del ds['C22']

    # Standard
    # resampled = warp(ds, extent=extent, resolution=resolution)
    # satio.to_netcdf(resampled, os.path.join(path, 'aligned', 'test.nc'))
    resampled = warp(ds, extent=extent, resolution=resolution, chunks=None,
                     precompute_coords=True)
    satio.to_netcdf(resampled,
                    os.path.join(path, 'aligned', 'test_chunked.nc'))


if __name__ == '__main__':
    main()
