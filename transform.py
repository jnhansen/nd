import numpy as np
import pandas as pd
import xarray as xr
import os, sys
import time
from osgeo import gdal, osr
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn import linear_model
import multiprocessing as mp

# from scipy.ndimage.interpolation import geometric_transform
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import map_coordinates

# --------------------------------------------------------------- #
# import resource
# rsrc = resource.RLIMIT_DATA
# soft, hard = resource.getrlimit(rsrc)
# print('Soft limit starts as  :', soft)
# resource.setrlimit(rsrc, (6*1024**3, hard))
# soft, hard = resource.getrlimit(rsrc)
# print('Soft limit changed to :', soft)
# --------------------------------------------------------------- #

try:
    type(profile)
except NameError:
    def profile(fn): return fn

# from mem_debug import debug_memory_leak

PY2 = sys.version_info < (3, 0)

NO_DATA_VALUE = -1

gdal.UseExceptions()

# ----------------------------------------------------------------------- #
import datetime as DT
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc


def str2date(string, fmt=None):
    if fmt is None:
        date_object = parsedate(string)
    else:
        date_object = DT.datetime.strptime(string, fmt)
    if date_object.tzinfo is None:
        date_object = date_object.replace(tzinfo=tzutc())
    return date_object
# ----------------------------------------------------------------------- #


def parallel_warp(src, output_shape=None):
    """
    """
    arr = src.ReadAsArray()
    gcps = src.GetGCPs()
    grid_y_interp, grid_x_interp = gcps2xygrid(gcps, (N_lat, N_lon))
    full_extent = latlon_extent(src)

    chunks = np.array_split(arr, mp.cpu_count())

    pool = mp.Pool()
    individual_results = pool.map(warp, chunks)
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def gcps2xygrid(gcps, output_shape, extent=None):
    attrs = ['GCPLine', 'GCPPixel', 'GCPX', 'GCPY', 'GCPZ']
    gcp_data = [{attr : getattr(gcp, attr) for attr in attrs} for gcp in gcps]
    gcp_df = pd.DataFrame(gcp_data)
    lon = gcp_df['GCPX']
    lat = gcp_df['GCPY']
    alt = gcp_df['GCPZ']

    ##
    ## Prepare the output data array (decide on resolution here).
    ##
    if extent is None:
        lat_range = [lat.min(), lat.max()]
        lon_range = [lon.min(), lon.max()]
    else:
        lon_range = extent[0::2]
        lat_range = extent[1::2]
    N_lat, N_lon = output_shape
    warped_grid_lon, warped_grid_lat = np.meshgrid(
        np.linspace(lon_range[0], lon_range[1], N_lon),
        np.linspace(lat_range[0], lat_range[1], N_lat)
    )

    ##
    ## The following grids should contain the (x,y) lookup for the original image.
    ##
    # method can be 'nearest', 'linear', or 'cubic'
    grid_y_interp = griddata(gcp_df[['GCPY', 'GCPX']], gcp_df['GCPLine'],
                             (warped_grid_lat, warped_grid_lon),
                             method='cubic')
    grid_x_interp = griddata(gcp_df[['GCPY', 'GCPX']], gcp_df['GCPPixel'],
                             (warped_grid_lat, warped_grid_lon),
                             method='cubic')
    return grid_y_interp, grid_x_interp


# def gcps2llgrid(gcps, output_shape):
#     attrs = ['GCPLine', 'GCPPixel', 'GCPX', 'GCPY', 'GCPZ']
#     gcp_data = [ { attr : getattr(gcp, attr) for attr in attrs } for gcp in gcps ]
#     gcp_df = pd.DataFrame(gcp_data)
#     lon = gcp_df['GCPX']
#     lat = gcp_df['GCPY']
#     alt = gcp_df['GCPZ']
#
#     lat_range = [lat.min(), lat.max()]
#     lon_range = [lon.min(), lon.max()]
#     N_lat, N_lon = output_shape
#     warped_grid_lon, warped_grid_lat = np.meshgrid(
#         np.linspace(lon_range[0], lon_range[1], N_lon),
#         np.linspace(lat_range[0], lat_range[1], N_lat)
#     )
#
#     grid_x, grid_y = np.mgrid[0:output_shape[0], 0:output_shape[1]]
#     ##
#     ## Interpolate the (lat,lon) coordinates for every pixel.
#     ##
#     # method can be 'nearest', 'linear', or 'cubic'
#     from scipy.interpolate import griddata
#     grid_lat = griddata(gcp_df[['GCPLine','GCPPixel']], lat, (grid_x, grid_y), method='linear')
#     grid_lon = griddata(gcp_df[['GCPLine','GCPPixel']], lon, (grid_x, grid_y), method='linear')
#
#     return grid_lat, grid_lon


@profile
def llgrid(src, output_shape=None, interpolate=False):
    """
    Generate a grid with the lat-lon coordinates of a GDAL dataset.

    Parameters
    ----------
    src : osgeo.gdal.DataSet
        The input data set.
    output_shape : tuple, opt
        The shape of the output raster (default: None).
        If None, infer from data.
    interpolate : bool, opt
        If True, fill the grid by interpolation from GCPs. Otherwise,
        fit a polynomial to lat(x,y) and lon(x,y). (default: False)

    Returns
    -------
    tuple of numpy.ndarray
        A tuple of the latitude grid and the longitude grid.
    """
    ##
    ## Use polynomial regression.
    ##
    gcp_df = get_gcp_df(src)
    lon = gcp_df['GCPX']
    lat = gcp_df['GCPY']
    if output_shape is None:
        N_lat, N_lon = src.RasterYSize, src.RasterXSize
    else:
        N_lat, N_lon = output_shape
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, src.RasterXSize-1, N_lon),
        np.linspace(0, src.RasterYSize-1, N_lat)
    )
    if interpolate:
        ##
        ## Interpolate the (lat,lon) coordinates for every pixel.
        ##
        # method can be 'nearest', 'linear', or 'cubic'
        grid_lat = griddata(gcp_df[['GCPLine', 'GCPPixel']], lat,
                            (grid_x, grid_y), method='linear')
        grid_lon = griddata(gcp_df[['GCPLine', 'GCPPixel']], lon,
                            (grid_x, grid_y), method='linear')
    else:
        xy2ll = latlon_fit(src, degree=2, inverse=False)
        coords = np.stack([grid_y.flatten(), grid_x.flatten()], axis=-1)
        ll = xy2ll(coords)
        grid_lat = ll[:, 0].reshape((N_lat, N_lon))
        grid_lon = ll[:, 1].reshape((N_lat, N_lon))
    return grid_lat, grid_lon


@profile
def warp(src, output_shape=None, extent=None, nproc=1, fake=False):
    """
    Warps a GDAL dataset onto a lat-lon grid (equirectangular projection).

    It is assumed that the dataset contains a list of GCPs for georeferencing.

    Parameters
    ----------
    src : osgeo.gdal.Dataset
    output_shape : tuple, opt
        The shape of the output raster (default: None).
        If None, infer from data.
    extent : list, opt
        The lat-lon extent of the output raster as
        [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat] (default: None).
        If None, infer from data.
    nproc : int, opt
        The number of parallel processes to use (default: 1).

    Returns
    -------
    osgeo.gdal.Dataset
        A warped dataset. The georeferencing is encoded in the transform matrix
        that can be accessed as `osgeo.gdal.Dataset.GetGeoTransform()`.
    """
    # data = src.ReadAsArray()
    # if len(data.shape) == 2:
    #     data = np.expand_dims(data, 0)
    # N_bands = data.shape[0]
    N_bands = src.RasterCount
    gcp_df = get_gcp_df(src)
    # points = gcp_df[['GCPLine','GCPPixel']]
    lon = gcp_df['GCPX']
    lat = gcp_df['GCPY']
    alt = gcp_df['GCPZ']

    ##
    ## Prepare the output data array (decide on resolution here).
    ##
    input_lat_range = [lat.min(), lat.max()]
    input_lon_range = [lon.min(), lon.max()]
    if extent is None:
        output_lat_range = input_lat_range
        output_lon_range = input_lon_range
    else:
        output_lon_range = extent[0::2]
        output_lat_range = extent[1::2]

    # input_lon_start = input_lon_range[0]
    # input_lat_start = input_lat_range[0]
    output_lon_start = output_lon_range[0]
    output_lat_start = output_lat_range[0]
    lonlat_step = 0
    latlon_step = 0
    if output_shape is None:
        N_lat, N_lon = src.RasterYSize, src.RasterXSize
    else:
        N_lat, N_lon = output_shape
    lat_step = (output_lat_range[1] - output_lat_range[0]) / (N_lat - 1)
    lon_step = (output_lon_range[1] - output_lon_range[0]) / (N_lon - 1)

    ##
    ## Interpolate the (lat,lon) coordinates for every pixel.
    ##
    # method can be 'nearest', 'linear', or 'cubic'
    # grid_x, grid_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    # grid_lat = griddata(gcp_df[['GCPLine','GCPPixel']], lat, (grid_x, grid_y), method='cubic')
    # grid_lon = griddata(gcp_df[['GCPLine','GCPPixel']], lon, (grid_x, grid_y), method='cubic')

    ##
    ## The following grids should contain the (x,y) lookup for the original image.
    ##
    # grid_y_interp, grid_x_interp = gcps2xygrid(gcps, (input_N_lat,input_N_lon))
    #
    # def coord_map(coords):
    #     # if coords[0] >= grid_y_interp.shape[0]
    #     return (grid_y_interp[coords], grid_x_interp[coords])

    ##
    ## Alternative approach:
    ## Regression instead of interpolation
    ##

    ## Map (lon,lat) to original (x,y)
    ## Map position on grid to (lon,lat)

    # start = np.array([output_lat_start, output_lon_start])
    # step = np.array([lat_step, lon_step])
    # def map_fn(coords):
    #     ll = start + step * np.array(coords)
    #     result = ll2xy(ll)
    #     if result.shape[0] == 1:
    #         return tuple(result[0])
    #     else:
    #         return result

    ll2xy = latlon_fit(src, degree=3, inverse=True)
    grid_ll = np.meshgrid(
        np.linspace(output_lon_range[0], output_lon_range[1], N_lon),
        np.linspace(output_lat_range[0], output_lat_range[1], N_lat),
        copy=False
    )
    ll = np.stack([grid_ll[1].flatten(), grid_ll[0].flatten()], axis=-1)
    coords = ll2xy(ll).T.reshape((2, N_lat, N_lon))

    # --------------------------------------------------------------------------- #
    # Trying a more efficient version of map_coordinates.
    # ll2xy = latlon_fit(src, degree=3, inverse=True)
    # offset = np.array([output_lon_range[0],output_lat_range[0]])
    # factor = np.array([lon_step, lat_step])
    # def coord_map(coords):
    #     ll = coords*factor + offset
    #     return ll2xy(ll)
    # --------------------------------------------------------------------------- #

    # warped = geometric_transform(data, map_fn, (N_lat,N_lon), output=np.float32, order=3, cval=NO_DATA_VALUE)

    ## DEBUGGING:
    # grid_lat_warped = geometric_transform(grid_lat, coord_map, (N_lat,N_lon), cval=NO_DATA_VALUE)
    # grid_lon_warped = geometric_transform(grid_lon, coord_map, (N_lat,N_lon), cval=NO_DATA_VALUE)

    ##
    ## Create GDAL dataset from numpy array.
    ##
    tmp = gdal.GetDriverByName('MEM').Create('', N_lon, N_lat, N_bands,
                                             gdal.GDT_Float32)

    if nproc == 1:
        warped = [map_coordinates(src.GetRasterBand(i+1).ReadAsArray(),
                                  coords, output=np.float32, order=3,
                                  cval=NO_DATA_VALUE) for i in range(N_bands)]
        # warped = [efficient_map_coordinates(src.GetRasterBand(i+1).ReadAsArray(), coord_map, output=np.float32, order=3, cval=NO_DATA_VALUE) for i in range(N_bands)]
    else:
        # warped = [parallel_map_coordinates(src.GetRasterBand(i+1).ReadAsArray(), coords, output=np.float32, order=3, cval=NO_DATA_VALUE, nproc=nproc, fake=fake) for i in range(N_bands)]
        kwargs_list = [dict(input=src.GetRasterBand(i+1).ReadAsArray(),
                            coordinates=coords, output=np.float32, order=3,
                            cval=NO_DATA_VALUE) for i in range(N_bands)]
        pool = mp.Pool(min(nproc, N_bands))
        result = pool.map_async(map_coordinates_wrapper, kwargs_list)
        pool.close()
        pool.join()
        warped = result.get()

    for i in range(N_bands):
        org_band = src.GetRasterBand(i+1)
        new_band = tmp.GetRasterBand(i+1)
        new_band.WriteArray(warped[i])
        new_band.SetNoDataValue(NO_DATA_VALUE)
        new_band.SetMetadata(org_band.GetMetadata())
    transform = (output_lon_start, lon_step, lonlat_step,
                 output_lat_start, latlon_step, lat_step)
    tmp.SetGeoTransform(transform)
    tmp.SetMetadata(src.GetMetadata())
    return tmp


# @profile
# def blockgen(array, bpa):
#     """
#     https://stackoverflow.com/a/16865342
#
#     Creates a generator that yields multidimensional blocks from the given
#     array(_like); bpa is an array_like consisting of the number of blocks per axis
#     (minimum of 1, must be a divisor of the corresponding axis size of array). As
#     the blocks are selected using normal numpy slicing, they will be views rather
#     than copies; this is good for very large multidimensional arrays that are being
#     blocked, and for very large blocks, but it also means that the result must be
#     copied if it is to be modified (unless modifying the original data as well is
#     intended).
#     """
#     bpa = np.asarray(bpa) # in case bpa wasn't already an ndarray
#
#     # parameter checking
#     if array.ndim != bpa.size:         # bpa doesn't match array dimensionality
#         raise ValueError("Size of bpa must be equal to the array dimensionality.")
#     if (bpa.dtype != np.int            # bpa must be all integers
#         or (bpa < 1).any()             # all values in bpa must be >= 1
#         or (array.shape % bpa).any()): # % != 0 means not evenly divisible
#         raise ValueError("bpa ({0}) must consist of nonzero positive integers "
#                          "that evenly divide the corresponding array axis "
#                          "size".format(bpa))
#
#     # generate block edge indices
#     rgen = (np.r_[:array.shape[i]+1:array.shape[i]//blk_n]
#             for i, blk_n in enumerate(bpa))
#
#     # build slice sequences for each axis (unfortunately broadcasting
#     # can't be used to make the items easy to operate over
#     c = [[np.s_[i:j] for i, j in zip(r[:-1], r[1:])] for r in rgen]
#
#     # Now to get the blocks; this is slightly less efficient than it could be
#     # because numpy doesn't like jagged arrays and I didn't feel like writing
#     # a ufunc for it.
#     for idxs in np.ndindex(*bpa):
#         blockbounds = tuple(c[j][idxs[j]] for j in range(bpa.size))
#         yield array[blockbounds]


def chunks(l, n):
    """Yield successive n-sized chunks from l.

    https://stackoverflow.com/a/312464

    Parameters
    ----------
    l : iterable
        The list or list-like object to be split into chunks.
    n : int
        The size of the chunks to be generated.

    Yields
    ------
    iterable
        Consecutive slices of l of size n.
    """
    if PY2:
        for i in xrange(0, len(l), n):
            yield l[i:i + n]
    else:
        for i in range(0, len(l), n):
            yield l[i:i + n]


def array_chunks(array, n, axis=0):
    if axis >= array.ndim:
        raise ValueError("axis {:d} is out of range for given array."
                         .format(axis))

    arr_len = array.shape[axis]
    range_fn = range if not PY2 else xrange
    for i in range_fn(0, arr_len, n):
        indices = [slice(None), ] * array.ndim
        indices[axis] = slice(i, i+n)
        yield array[indices]


# def blockmerge(array, bpa):
#     """
#     Reassemble a list of arrays as generated by blockgen.
#     """
#     if len(array) != np.prod(bpa):
#         raise ValueError("Length of array must be equal to the product of "
#                          "the shape elements.")
#     # if array.ndim != len(bpa):
#     #     raise ValueError("Size of bpa must be equal to the array "
#     #                      "dimensionality.")
#
#     result = array
#     for i, l in enumerate(bpa[::-1]):
#         result = np.concatenate([_ for _ in chunks(result, l)],
#                                 axis=len(bpa)-i-1)
#         # return np.concatenate([np.concatenate(_, axis=1) for _ in
#         #                        chunks(array, bpa[1])], axis=0)
#     return result


def block_split(array, blocks):
    """
    Split an ndarray into subarrays according to blocks.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be split.
    blocks : array_like
        The desired number of blocks per axis.

    Returns
    -------
    list
        A list of blocks, in column-major order.

    Examples
    --------
    >>> block_split(np.arange(16).reshape((4,4)))
    [array([[ 0,  1],
            [ 4,  5]]),
     array([[ 2,  3],
            [ 6,  7]]),
     array([[ 8,  9],
            [12, 13]]),
     array([[10, 11],
            [14, 15]])]

    """
    if array.ndim != len(blocks):
        raise ValueError("Length of 'blocks' must be equal to the "
                         "array dimensionality.")

    result = [array]
    for axis, nblocks in enumerate(blocks):
        result = [np.array_split(_, nblocks, axis=axis) for _ in result]
        result = [item for sublist in result for item in sublist]
    return result


def block_merge(array_list, blocks):
    """
    Reassemble a list of arrays as generated by block_split.

    Parameters
    ----------
    array_list : list of numpy.array
        A list of numpy.array, e.g. as generated by block_split().
    blocks : array_like
        The number of blocks per axis to be merged.

    Returns
    -------
    numpy.array
        A numpy array with dimension len(blocks).

    """
    if len(array_list) != np.prod(blocks):
        raise ValueError("Length of array list must be equal to the "
                         "product of the shape elements.")

    result = array_list
    for i, nblocks in enumerate(blocks[::-1]):
        axis = len(blocks) - i - 1
        result = [np.concatenate(_, axis=axis)
                  for _ in chunks(result, nblocks)]
    return result[0]


# def efficient_map_coordinates(input, coord_map, output_shape=None,
#                               output=np.float32, **kwargs):
#     ## Generate output array.
#     if output_shape is None:
#         output_shape = input.shape
#     result = np.empty(output_shape)
#     result[:] = cval
#     map_coordinates(src.GetRasterBand(i+1).ReadAsArray(), coords,
#                     output=np.float32, order=3, cval=NO_DATA_VALUE)
#     pass


def map_coordinates_wrapper(kwargs):
    from scipy.ndimage.interpolation import map_coordinates
    return map_coordinates(**kwargs)


@profile
def parallel_map_coordinates(input, coordinates, nproc=1, fake=False,
                             **kwargs):
    """
    A parallel version of scipy.ndimage.interpolation.map_coordinates().
    """
    # coordinate_chunks = np.array_split(coordinates, nproc, axis=1)
    blocks = (2,2)
    coordinate_chunks = [_ for _ in block_split(coordinates, (1,)+blocks)]
    ##
    ## Possibly cut input to only cover [chunk.min(),chunk.max()]
    ##
    # ...
    input_chunks = []
    for c in coordinate_chunks:
        cmin = np.floor(c.min(axis=1).min(axis=1)).astype(int)
        cmax = np.ceil(c.max(axis=1).max(axis=1)).astype(int)
        cmin[cmin < 0] = 0
        cmax[cmax > input.shape] = np.array(input.shape)[cmax > input.shape]
        i = input[cmin[0]:cmax[0]+1, cmin[1]:cmax[1]+1]
        input_chunks.append(i)
        c -= np.expand_dims(np.expand_dims(cmin, axis=-1), axis=-1)
        # print('c', c.shape)
        # print('i', i.shape)
        # print('i', np.prod(i.shape))
        # print('coord bounds', cmin, cmax)

    kwargs_chunks = [dict(input=ichunk, coordinates=cchunk, **kwargs)
                     for cchunk, ichunk in zip(coordinate_chunks,
                                               input_chunks)]
    # return
    ## DEBUG: in serial
    if fake:
        individual_results = [map_coordinates_wrapper(chunk)
                              for chunk in kwargs_chunks]
        # return np.concatenate(individual_results)
        return block_merge(individual_results, blocks)
    else:
        pool = mp.Pool(nproc)
        individual_results = pool.map_async(map_coordinates_wrapper,
                                            kwargs_chunks)
        pool.close()
        pool.join()
        # from IPython import embed; embed()
        return block_merge(individual_results.get(), blocks)


def warp_to_4326(src):
    """ Warps a GDAL dataset onto EPSG:4326, i.e. a lat-lon grid.
    https://gis.stackexchange.com/a/140053

    This is deprecated and replaced by warp().

    Parameters
    ----------
    src : osgeo.gdal.Dataset

    Returns
    -------
    osgeo.gdal.Dataset
        A dataset warped onto EPSG:4326 (lat-lon grid)
    """
    # src_ds = gdal.Open(path)

    # Define target SRS
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    dst_wkt = dst_srs.ExportToWkt()

    error_threshold = 0.125
    resampling = gdal.GRA_Bilinear
    # resampling = gdal.GRA_NearestNeighbour

    warped_ds = gdal.AutoCreateWarpedVRT(src,
                                         None,  # src_wkt : left to default value --> will use the one from source
                                         dst_wkt,
                                         resampling,
                                         error_threshold)

    return warped_ds


def map_coordinates_with_nan(input, coords, *args, **kwargs):
    """
    An extension of map_coordinates that can handle np.nan values in the
    input array.
    """
    nanmask = np.isnan(input).astype(np.float32)
    if nanmask.sum() > 0:
        nanmask_mapped = map_coordinates(nanmask, coords, output=np.float32,
                                         cval=1) > 0.9
    else:
        nanmask_mapped = np.zeros_like(coords)
    filled_input = input.copy()
    filled_input[np.isnan(filled_input)] = 0
    result = map_coordinates(filled_input, coords, *args, **kwargs)
    result[nanmask_mapped] = np.nan
    return result


def resample_grid(dataset, shape, extent):
    """
    Parameters
    ----------
    dataset : xarray.Dataset
    shape : tuple
    extent : array_like
    """
    new_coords = dict(dataset.coords)
    new_coords['lat'] = np.linspace(extent[0], extent[2], shape[0])
    new_coords['lon'] = np.linspace(extent[1], extent[3], shape[1])

    lon_grid, lat_grid = np.meshgrid(new_coords['lon'], new_coords['lat'],
                                     copy=False)
    latlon_grid = np.stack([lat_grid, lon_grid], axis=0)

    org_ll_min = np.array([[[dataset.lat.min()]], [[dataset.lon.min()]]])
    org_ll_range = np.array([[[dataset.lat.max()]],
                             [[dataset.lon.max()]]]) - org_ll_min
    org_ll_shape = np.array([[[dataset.sizes['lat']]],
                             [[dataset.sizes['lon']]]]) - 1
    coord_grid = (latlon_grid - org_ll_min) * (org_ll_shape/org_ll_range)

    result = xr.Dataset(coords=new_coords, attrs=dataset.attrs)
    for var in dataset.data_vars:
        result[var] = (('lat', 'lon'),
                       map_coordinates_with_nan(dataset[var].values,
                                                coord_grid, output=np.float32,
                                                order=2, cval=np.nan))

    return result


def align_and_merge(src_datasets, resolution=None, mode='union'):
    """
    Align multiple datasets to a common grid.

    Parameters
    ----------
    src_datasets : list of xarray.dataset
        The datasets to be aligned.
    resolution : tuple of float, optional
        The resolution (in degrees) of lat and lon. If not passed, determine
        resolution from data.
    mode : str, optional
        One of 'union' or 'intersect'. If 'union', creates a dataset that
        covers

    Returns
    -------
    xarray.dataset
        A single merged dataset on a common grid.
    """
    extents = []
    resolutions = []
    for x in src_datasets:
        extent = np.array([x.lat.min(), x.lon.min(), x.lat.max(), x.lon.max()])
        extents.append(extent)
        resolutions.append(np.array([(extent[2] - extent[0]) / x.sizes['lat'],
                                     (extent[3] - extent[1]) / x.sizes['lon']])
                           )

    extents = np.array(extents)
    resolutions = np.array(resolutions)
    new_extent = np.concatenate([extents[:, :2].min(axis=0),
                                 extents[:, 2:].max(axis=0)])
    new_resolution = resolutions.min(axis=0)
    new_shape = ((new_extent[2:] - new_extent[:2]) /
                 new_resolution).astype(int)
    new_lats = np.arange(new_extent[0], new_extent[2], new_resolution[0])
    new_lons = np.arange(new_extent[1], new_extent[3], new_resolution[1])
    # new_coords = {'lat':new_lats,'lon':new_lons}

    new_datasets = [resample_grid(x, shape=new_shape, extent=new_extent)
                    for x in src_datasets]
    res = xr.concat(new_datasets, dim='time').sortby('time')
    return res


def to_xr(src):
    """
    Convert GDAL dataset to xarray dataset.

    Parameters
    ----------
    src : osgeo.gdal.Dataset

    Returns
    -------
    xarray.Dataset
    """
    x, y = src.RasterXSize, src.RasterYSize
    meta = src.GetMetadata()
    times = [str2date(meta['ACQUISITION_START_TIME'])]
    ##
    ## Determine whether the dataset uses GCPs or GeoTransform.
    ##
    if src.GetGCPCount() > 0:
        lat, lon = llgrid(src)
        ds = xr.Dataset(coords={'lat': (['x', 'y'], lat),
                                'lon': (['x', 'y'], lon),
                                'time': times})
        data_coords = ('y', 'x')
    else:
        transform = src.GetGeoTransform()
        lon_start, lon_step, _, lat_start, _, lat_step = transform
        lats = lat_start + np.arange(y) * lat_step
        lons = lon_start + np.arange(x) * lon_step
        ds = xr.Dataset(coords={'lat': lats, 'lon': lons, 'time': times},
                        attrs=meta)
        data_coords = ('lat', 'lon')

    for i in range(src.RasterCount):
        band = src.GetRasterBand(i+1)
        ndv = band.GetNoDataValue()
        name = band.GetMetadata()['POLARISATION']
        data = band.ReadAsArray()
        data[data == ndv] = np.nan
        ds[name] = (data_coords, data)

    return ds


def merge_datasets(datasets):
    """
    Merge a list of xarray datasets.
    """
    ##
    ## Step 1. Regrid onto common lat-lon grid.
    ##
    # Determine individual resolutions. Store as metadata?
    #

    ##
    ## Step 2. Concatenate along time axis.
    ##


def scaled_gcps(src, output_shape):
    """
    Given a source dataset, scale the available GCPs to match the specified
    output shape.

    Parameters
    ----------
    src : osgeo.gdal.Dataset
        The dataset, including GCPs.
    output_shape : tuple
        The shape that the GCPs need to be scaled to.

    Returns
    -------
    tuple of osgeo.gdal.GCP
        A tuple of GCPs, in the same order as the original GCPs, with scaled
        `Line` and `Pixel` coordinates.
    """
    gcps = src.GetGCPs()
    input_shape = (src.RasterYSize, src.RasterXSize)
    factor_y = float(output_shape[0]) / input_shape[0]
    factor_x = float(output_shape[1]) / input_shape[1]
    new_gcps = tuple(gdal.GCP(gcp.GCPX, gcp.GCPY, gcp.GCPZ,
                              gcp.GCPPixel*factor_x, gcp.GCPLine*factor_y,
                              gcp.Info, gcp.Id) for gcp in gcps)
    return new_gcps


def latlon_extent(src):
    """
    Given a source dataset, return the lat-lon extent.

    Parameters
    ----------
    src : osgeo.gdal.Dataset
        The source dataset.

    Returns
    -------
    list
        The extent as [min_lon,min_lat,max_lon,max_lat].

    """
    transform = src.GetGeoTransform()
    x, y = src.RasterXSize, src.RasterYSize
    lon_start, lon_step, lonlat_step, \
        lat_start, latlon_step, lat_step = transform

    lon_ur = lon_start + x*lon_step + 0*lonlat_step
    lon_lr = lon_start + x*lon_step + y*lonlat_step
    lon_ul = lon_start + 0*lon_step + 0*lonlat_step
    lon_ll = lon_start + 0*lon_step + y*lonlat_step

    lat_ur = lat_start + 0*lat_step + x*latlon_step
    lat_ul = lat_start + 0*lat_step + 0*latlon_step
    lat_lr = lat_start + y*lat_step + x*latlon_step
    lat_ll = lat_start + y*lat_step + 0*latlon_step

    # extent = [lon_ll, lat_ll, lon_ur, lat_ur]
    extent = [
        min(lon_ur, lon_lr, lon_ul, lon_ll),
        min(lat_ur, lat_lr, lat_ul, lat_ll),
        max(lon_ur, lon_lr, lon_ul, lon_ll),
        max(lat_ur, lat_lr, lat_ul, lat_ll)
    ]
    return extent


def get_gcp_df(src):
    """
    Converts the GCPs from the source dataset into a pandas.DataFrame.

    Parameters
    ----------
    src : osgeo.gdal.Dataset
        The dataset, including GCPs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns
        `['GCPLine', 'GCPPixel', 'GCPX', 'GCPY', 'GCPZ']`

    """
    gcps = src.GetGCPs()
    attrs = ['GCPLine', 'GCPPixel', 'GCPX', 'GCPY', 'GCPZ']
    gcp_data = [{attr: getattr(gcp, attr) for attr in attrs} for gcp in gcps]
    return pd.DataFrame(gcp_data)


def gcp_extent(src):
    gcp_df = get_gcp_df(src)
    lon = gcp_df['GCPX']
    lat = gcp_df['GCPY']
    alt = gcp_df['GCPZ']
    extent = [lon.min(), lat.min(), lon.max(), lat.max()]
    return extent


def read_tiffs(tiffs):
    """
    Deprecate.

    Read multiple GeoTIFFs and warp onto a common lat-lon grid such that the
    shape and extent of each output dataset is the same.

    Parameters
    ----------
    tiffs : list of str
        A list of the filepaths of the GeoTIFFs to be read.

    Returns
    -------
    list of osgeo.gdal.Dataset
        A list of the warped datasets.
    """
    datasets = [gdal.Open(t) for t in tiffs]
    extents = np.array([gcp_extent(d) for d in datasets])
    combined_extent = [
        extents[:, 0].min(),
        extents[:, 1].min(),
        extents[:, 2].max(),
        extents[:, 3].max(),
    ]
    warped = [warp(d, output_shape=(1000, 1000), extent=combined_extent)
              for d in datasets]

    return warped


@profile
def latlon_fit(src, degree=2, inverse=False):
    """
    Fit a polynomial to the Ground Control Points (GCPs) of the source dataset.

    Parameters
    ----------
    src : gdal.DataSet
    degree : int, opt
        The polynomial degree to be fitted (default: 2)
    inverse : bool, opt
        If True, fit x,y as function of lat,lon (default: False).

    Returns
    -------
    function
        If `inverse` is False, the returned function converts (y,x) to
        (lat,lon). Otherwise, the function returns (lat,lon) for an input of
        (y,x).

    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    df = get_gcp_df(src)
    poly = PolynomialFeatures(degree=degree)
    if inverse:
        regressor = poly.fit_transform(df[['GCPY', 'GCPX']])
        regressand = df[['GCPLine', 'GCPPixel']]
    else:
        regressor = poly.fit_transform(df[['GCPLine', 'GCPPixel']])
        regressand = df[['GCPY', 'GCPX']]
    clf = linear_model.LinearRegression()
    clf.fit(regressor, regressand)

    def fn(X):
        """
        This function maps from (x,y) to (lat,lon) (or the reverse).
        """
        if not type(X) is np.ndarray or not len(X.shape) == 2:
            X = np.array([X])
        ##
        ## If X is very large, split into chunks.
        ##
        # Empirical optimal chunk sizes (approximately):
        # chunk size | len(X) | degree
        # -----------|--------|--------
        # 8000       | 1e8    | 3
        # 8000       | 1e8    | 2
        res = []
        for chunk in array_chunks(X, 8000, axis=0):
            p = poly.transform(chunk)
            res.append(clf.predict(p))
        return np.concatenate(res)

    return fn


def read_S1_GRD(path):
    N = 10
    data_path = os.path.join(path, 'measurement')
    tiffs = [os.path.join(data_path, _) for _ in os.listdir(data_path)
             if _.endswith('.tiff')]
    ## GET GEO REFERENCE
    gdata = gdal.Open(tiffs[0])
    gcp_df = get_gcp_df(gdata)
    ## INTERPOLATE
    poly = PolynomialFeatures(degree=2)
    regressor = poly.fit_transform(gcp_df[['GCPLine', 'GCPPixel']])
    regressand = gcp_df[['GCPX', 'GCPY']]
    clf = linear_model.LinearRegression()
    clf.fit(regressor, regressand)
    ## SCORE INTERPOLATION
    # clf.score(regressor,regressand)
    # from sklearn.metrics import r2_score
    # pred = clf.predict(regressor)
    # r2_score(regressand,pred)

    ## RESAMPLE
    xs = range(gdata.RasterXSize)[::N]
    ys = range(gdata.RasterYSize)[::N]
    xygrid = np.stack(np.meshgrid(xs, ys), axis=-1)

    ## GENERATE LAT LON GRID
    xygrid_flat = xygrid.reshape((-1, 2))
    llgrid_flat = clf.predict(poly.fit_transform(xygrid_flat))
    ll_grid = llgrid_flat.reshape(xygrid.shape[:2] + llgrid_flat.shape[-1:])
    result = ll_grid

    for tiff in tiffs:
        ## OPEN FILE
        gdata = gdal.Open(tiff)
        data = gdata.ReadAsArray()
        ## RESAMPLE
        data_ = data[::N, ::N]
        ## APPEND TO RESULT
        result = np.concatenate([result,
                                 np.expand_dims(data_, axis=2)], axis=2)

    return result

# ------------------------------------------------------------------------ #
# DEBUGGING
# ------------------------------------------------------------------------ #
import linecache


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


if __name__ == '__main__':
    # import tracemalloc
    # tracemalloc.start(25)

    path = '/Users/jhansen/ED/indonesia/S1A_IW_GRDH_1SDV_20180111T225606_20180111T225631_020113_0224BA_A598.SAFE/manifest.safe'
    src = gdal.Open(path)
    output_shape = (10000, 10000)
    # output_shape=None
    # lat,lon = llgrid(src, output_shape)

    # t1 = time.time()
    warped = warp(src, output_shape=output_shape)
    # t2 = time.time()
    # print('serial:        {:.1f}s'.format(t2-t1))
    # warped_parallel = warp(src, output_shape=output_shape, nproc=mp.cpu_count())
    # t3 = time.time()
    # print('parallel:      {:.1f}s'.format(t3-t2))
    # warped_parallel_fake = warp(src, output_shape=output_shape, nproc=mp.cpu_count(), fake=True)
    # t4 = time.time()
    # print('fake parallel: {:.1f}s'.format(t4-t3))

    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)
    # from IPython import embed; embed()
