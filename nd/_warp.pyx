# cython: profile=True
cimport cython
import numpy as np
cimport numpy as np
from cython cimport floating
# from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport isnan, abs
import xarray as xr
from scipy.ndimage.interpolation import map_coordinates, affine_transform


ctypedef np.float64_t DOUBLE
ctypedef np.float32_t FLOAT
ctypedef Py_ssize_t SIZE_TYPE
ctypedef np.uint8_t BOOL

ctypedef fused data_type:
    double
    float
ctypedef fused coord_type:
    double
    float


cdef class CoordTransform:
    cdef:
        double o_lon_ll, o_lat_ll, o_lon_ur, o_lat_ur
        Py_ssize_t o_lat_size, o_lon_size
        double lon_ll, lat_ll, lon_ur, lat_ur
        Py_ssize_t lat_size, lon_size
        double o_lat_range, o_lon_range
        double lat_range, lon_range
        double o_lat_density, o_lon_density
        double lat_density, lon_density

    def __cinit__(self,
                  tuple o_extent,
                  tuple o_shape,
                  tuple extent,
                  tuple shape):
        self.o_lon_ll, self.o_lat_ll, self.o_lon_ur, self.o_lat_ur = o_extent
        self.o_lat_size, self.o_lon_size = o_shape
        self.lon_ll, self.lat_ll, self.lon_ur, self.lat_ur = extent
        self.lat_size, self.lon_size = shape

        self.o_lat_range = self.o_lat_ur - self.o_lat_ll
        self.o_lon_range = self.o_lon_ur - self.o_lon_ll

        self.lat_range = self.lat_ur - self.lat_ll
        self.lon_range = self.lon_ur - self.lon_ll

        self.o_lat_density = float(self.o_lat_size) / self.o_lat_range
        self.o_lon_density = float(self.o_lon_size) / self.o_lon_range

        self.lat_density = float(self.lat_size) / self.lat_range
        self.lon_density = float(self.lon_size) / self.lon_range

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef tuple apply(self, tuple c):
        cdef:
            double lat, lon
            double y, x
            Py_ssize_t i_lat, i_lon
        i_lat, i_lon = c
        lat = self.lat_ur + i_lat * self.lat_density
        lon = self.lon_ll + i_lon * self.lon_density
        y = (self.o_lat_ur - lat) * self.o_lat_density
        x = (lon - self.o_lon_ll) * self.o_lon_density
        return (y, x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float[:, :, :] c_grid(tuple o_extent,
                            tuple o_shape,
                            tuple extent,
                            tuple shape):
    """
    Generate a grid of image coordinates based on original and new extent,
    as well as original and new shape or resolution.

    Parameters
    ----------
    o_extent : tuple
        The original extent as (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat).
    o_shape : tuple
        The original shape as (rows, cols).
    extent : tuple
        The new extent as (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat).
    shape : tuple
        The new shape as (rows, cols).

    Returns
    -------
    double [:, :, :], shape (2, shape[0], shape[1])
        A coordinate array that can be passed e.g. to map_coordinates to
        interpolate the corresponding values.
    """
    cdef:
        double lon_ll, lat_ll, lon_ur, lat_ur
        double o_lon_ll, o_lat_ll, o_lon_ur, o_lat_ur
        double lon_res, lat_res
        SIZE_TYPE lat_size, lon_size
        SIZE_TYPE o_lat_size, o_lon_size
        double o_lat_range, o_lon_range
        double o_lat_density, o_lon_density
        SIZE_TYPE i_lat, i_lon
        float [:, :, :] coord_grid
        double [:] new_coords_lat, new_coords_lon
        float [:] new_coords_y, new_coords_x

    lon_ll = extent[0]
    lat_ll = extent[1]
    lon_ur = extent[2]
    lat_ur = extent[3]
    lat_size = shape[0]
    lon_size = shape[1]
    o_lon_ll = o_extent[0]
    o_lat_ll = o_extent[1]
    o_lon_ur = o_extent[2]
    o_lat_ur = o_extent[3]
    o_lat_size = o_shape[0]
    o_lon_size = o_shape[1]

    o_lat_range = o_lat_ur - o_lat_ll
    o_lon_range = o_lon_ur - o_lon_ll

    o_lat_density = float(o_lat_size) / o_lat_range
    o_lon_density = float(o_lon_size) / o_lon_range

    new_coords_lat = np.linspace(lat_ur, lat_ll, lat_size)
    new_coords_lon = np.linspace(lon_ll, lon_ur, lon_size)

    new_coords_y = np.empty(lat_size, dtype=np.float32)
    new_coords_x = np.empty(lon_size, dtype=np.float32)
    for i_lat in range(lat_size):
        new_coords_y[i_lat] = \
            (o_lat_ur - new_coords_lat[i_lat]) * o_lat_density
            # (new_coords_lat[i_lat] - o_lat_ll) * o_lat_density
    for i_lon in range(lon_size):
        new_coords_x[i_lon] = \
            (new_coords_lon[i_lon] - o_lon_ll) * o_lon_density

    # Generate coordinate grid
    coord_grid = np.empty((2, lat_size, lon_size), dtype=np.float32)

    for i_lat in range(lat_size):
        for i_lon in range(lon_size):
            coord_grid[0, i_lat, i_lon] = new_coords_y[i_lat]
            coord_grid[1, i_lat, i_lon] = new_coords_x[i_lon]

    return coord_grid


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple c_valid(coord_type[:, :, :] coords,
                    tuple shape):
    """
    Given a coordinate array (generated e.g. from `c_grid`) and the shape of
    the original data, return a flattened (2-dimensional) coordinate array
    with only valid coordinates and a boolean mask corresponding to the indices
    of the valid coordinates.

    Parameters
    ----------
    coords : coord_type [:, :, :], shape (2, M, N)
        A coordinate array that can be passed e.g. to map_coordinates to
        interpolate the corresponding values.
    shape : tuple (int, int)
        The shape of the original image. Coordinates exceeding this shape are
        considered invalid.

    Returns
    -------
    tuple(np.ndarray[BOOL, ndim=2], np.ndarray[coord_type, ndim=2])
        A boolean mask for valid coordinates, and the flattened valid
        coordinates.
    """
    cdef:
        SIZE_TYPE o_lat_size, o_lon_size
        SIZE_TYPE lat_size, lon_size
        SIZE_TYPE i_lat, i_lon
        SIZE_TYPE counter
        coord_type c_lat, c_lon
        np.ndarray[coord_type, ndim=2] valid_coords
        np.ndarray[BOOL, ndim=2] out_of_range

    # Original shape
    o_lat_size = shape[0]
    o_lon_size = shape[1]
    # Shape of coordinate array
    lat_size = coords.shape[1]
    lon_size = coords.shape[2]
    out_of_range = np.empty((lat_size, lon_size), dtype=np.uint8)
    c_dtype = np.float32 if coord_type is float else np.float64
    valid_coords = np.empty((2, lat_size * lon_size), dtype=c_dtype)
    counter = 0
    for i_lat in range(lat_size):
        for i_lon in range(lon_size):
            c_lat = coords[0, i_lat, i_lon]
            c_lon = coords[1, i_lat, i_lon]
            if c_lat > o_lat_size \
                    or c_lon > o_lon_size \
                    or c_lat < 0 \
                    or c_lon < 0:
                out_of_range[i_lat, i_lon] = 1
            else:
                out_of_range[i_lat, i_lon] = 0
                valid_coords[0, counter] = c_lat
                valid_coords[1, counter] = c_lon
                counter += 1

    valid_coords = valid_coords[:, :counter]

    return (out_of_range.view(bool),
            valid_coords)



# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# @cython.cdivision(True)
# cpdef c_resample_grid(dataset, extent, shape=None, resolution=None):
#     """Resample a dataset to a new extent.

#     Parameters
#     ----------
#     dataset : xarray.Dataset
#         The dataset to resample.
#     extent : array_like
#         The extent of the output dataset.
#     shape : tuple, optional
#         The shape of the output dataset.
#         Ignored if `resolution` is also passed.
#     resolution : tuple, optional
#         The resolution of the output dataset.

#     Returns
#     -------
#     xarray.Dataset
#         A new dataset with shape `shape` and extent `extent`.
#     """
#     cdef:
#         double lon_min, lat_min, lon_max, lat_max
#         double lon_res, lat_res
#         double c_lat, c_lon
#         SIZE_TYPE lat_size, lon_size
#         SIZE_TYPE o_lat_size, o_lon_size
#         SIZE_TYPE i_lat, i_lon
#         SIZE_TYPE counter
#         np.ndarray[DOUBLE, ndim=3] coord_grid
#         np.ndarray[DOUBLE, ndim=2] coords_flat
#         np.ndarray[DOUBLE, ndim=2] valid_coords
#         np.ndarray[BOOL, ndim=2] out_of_range
#         # np.ndarray[DOUBLE, ndim=1] _resampled
#         # np.ndarray[DOUBLE, ndim=2] resampled
#         # np.ndarray[DOUBLE, ndim=1] new_coords_lat, new_coords_lon
#         # np.ndarray[DOUBLE, ndim=2] _values

#     lon_min, lat_min, lon_max, lat_max = extent
#     o_lat_size = dataset.sizes['lat']
#     o_lon_size = dataset.sizes['lon']
#     if resolution is not None:
#         lon_res, lat_res = resolution
#         # Set shape (a passed shape will be ignored).
#         lon_size = int((lon_max - lon_min) / lon_res)
#         lat_size = int((lat_max - lat_min) / lat_res)
#     elif shape is None:
#         # Automatically determine shape?
#         raise ValueError("Need to pass either shape or resolution!")
#     else:
#         lat_size, lon_size = shape

#     o_extent = (dataset.lon.values.min(), dataset.lat.values.min(),
#                 dataset.lon.values.max(), dataset.lat.values.max())

#     print('Calculating coord grid')
#     coord_grid = c_grid(o_extent, tuple((o_lat_size, o_lon_size)),
#                         tuple(extent), tuple((lat_size, lon_size)))

#     new_coords = dict(dataset.coords)
#     new_coords_lat = np.linspace(lat_min, lat_max, lat_size)
#     new_coords_lon = np.linspace(lon_min, lon_max, lon_size)
#     new_coords['lat'] = new_coords_lat
#     new_coords['lon'] = new_coords_lon
#     result = xr.Dataset(coords=new_coords, attrs=dataset.attrs)

#     # Exclude all coordinates that extend beyond the coordinate
#     # range of the original data.
#     print('Calculating out of range coords')
#     out_of_range = np.empty((lat_size, lon_size), dtype=np.uint8)
#     coords_flat = np.empty((2, lat_size * lon_size))
#     counter = 0
#     for i_lat in range(lat_size):
#         for i_lon in range(lon_size):
#             c_lat = coord_grid[0, i_lat, i_lon]
#             c_lon = coord_grid[1, i_lat, i_lon]
#             if c_lat > o_lat_size \
#                     or c_lon > o_lon_size \
#                     or c_lat < 0 \
#                     or c_lon < 0:
#                 out_of_range[i_lat, i_lon] = 1
#                 pass
#             else:
#                 out_of_range[i_lat, i_lon] = 0
#                 coords_flat[0, counter] = c_lat
#                 coords_flat[1, counter] = c_lon
#                 counter += 1

#     print('Calculating valid coords')
#     print(type(coords_flat), coords_flat.dtype,
#           coords_flat.shape[0], coords_flat.shape[1])
#     valid_coords = np.empty((2, counter))
#     valid_coords = coords_flat[:, :counter]
#     print(type(valid_coords), valid_coords.dtype,
#           valid_coords.shape[0], valid_coords.shape[1])

#     # valid_coords = coord_grid.transpose((1, 2, 0))[~out_of_range].transpose()

#     print('Looping through vars')
#     for var in dataset.data_vars:
#         if 'lat' not in dataset[var].coords or \
#                 'lon' not in dataset[var].coords:
#             continue

#         print('=== var: ', var)
#         print('resampled = ...')
#         resampled = np.full((lat_size, lon_size), np.nan)
#         print('_values = ...')
#         _values = dataset[var].values
#         print('map_coordinates_with_nan')
#         print(type(_values), _values.dtype, _values.shape)
#         _resampled = map_coordinates_with_nan(_values, valid_coords,
#                                               order=2, cval=np.nan)
#         print('resampled[~out_of_range] = ...')
#         resampled[~out_of_range.astype(bool)] = _resampled
#         print('result[var] = ...')
#         result[var] = (('lat', 'lon'), resampled)

#     return result


# Specialized function
# cpdef np.ndarray[data_type, ndim=1] _resample_grid(
#         np.ndarray[data_type, ndim=2] values,
#         np.ndarray[coord_type, ndim=2] coords,
#         int order,
#         data_type cval):
#     cdef:
#         np.ndarray[data_type, ndim=1] result

#     resampled = np.full((lat_size, lon_size), np.nan)
#     print('map_coordinates_with_nan')
#     print(type(_values), _values.dtype, _values.shape)
#     _resampled = map_coordinates_with_nan(
#         values, coords, order=order, cval=cval)
#     print('resampled[~out_of_range] = ...')
#     resampled[~out_of_range.astype(bool)] = _resampled
#     print('result[var] = ...')
#     result[var] = (('lat', 'lon'), resampled)

# def map_wrapper(*args, **kwargs):
#     return map_coordinates_with_nan(*args, **kwargs)


#
# MAP_COORDINATES FOR FLAT COORDINATE ARRAY
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cpdef np.ndarray[floating, ndim=1] map_coordinates_with_nan(
#         np.ndarray[floating, ndim=2] data,
#         np.ndarray[DOUBLE, ndim=2] coords,
#         int order=2,
#         float cval=np.nan):
#     """
#     An extension of map_coordinates that can handle np.nan values in the
#     data array.
#     """
#     cdef:
#         SIZE_TYPE o_lat_size, o_lon_size
#         SIZE_TYPE result_size
#         SIZE_TYPE i, i_lat, i_lon
#         np.ndarray[floating, ndim=2] filled_data
#         np.ndarray[floating, ndim=1] result
#         np.ndarray[FLOAT, ndim=2] nanmask
#         np.ndarray[FLOAT, ndim=1] nanmask_mapped
#         float nan_sum = 0
#     print(':: using map_coordinates for FLAT coords')

#     o_lat_size = data.shape[0]
#     o_lon_size = data.shape[1]
#     print('init nanmask')
#     nanmask = np.empty((o_lat_size, o_lon_size), dtype=np.float32)
#     print('copy data')
#     filled_data = data.copy()

#     print('fill nanmask')
#     #
#     # Wherever `data` is NaN, set a 0 in `filled_data` and set a 1 in
#     # `nanmask`. `nanmask` will then be warped itself to find NaN values in the
#     # warped image.
#     #
#     for i_lat in range(o_lat_size):
#         for i_lon in range(o_lon_size):
#             if isnan(data[i_lat, i_lon]):
#                 nanmask[i_lat, i_lon] = 1
#                 filled_data[i_lat, i_lon] = 0
#                 nan_sum += 1
#             else:
#                 nanmask[i_lat, i_lon] = 0

#     print('map_coordinates filled_data')
#     result = map_coordinates(filled_data, coords,
#                              order=order, cval=cval)
#     result_size = len(result)

#     print('map_coordinates nanmask')
#     if nan_sum > 0:
#         nanmask_mapped = map_coordinates(nanmask, coords, order=order, cval=1)
#         for i in range(result_size):
#             if nanmask_mapped[i] > 0.9:
#                 result[i] = np.nan

#     print('return result')

#     return result


# #
# # MAP_COORDINATES FOR 2D COORDINATE ARRAY
# #
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef np.ndarray[floating, ndim=2] map_coordinates_with_nan(
#         np.ndarray[floating, ndim=2] data,
#         np.ndarray[DOUBLE, ndim=3] coords,
#         int order=2,
#         float cval=np.nan):
#     """
#     An extension of map_coordinates that can handle np.nan values in the
#     data array.
#     """
#     cdef:
#         SIZE_TYPE o_lat_size, o_lon_size
#         SIZE_TYPE lat_size, lon_size
#         SIZE_TYPE i_lat, i_lon
#         np.ndarray[floating, ndim=2] filled_data
#         np.ndarray[floating, ndim=2] result
#         np.ndarray[FLOAT, ndim=2] nanmask
#         np.ndarray[FLOAT, ndim=2] nanmask_mapped
#         float nan_sum = 0

#     print(':: using map_coordinates for 2D coords')

#     o_lat_size = data.shape[0]
#     o_lon_size = data.shape[1]
#     print('init nanmask')
#     nanmask = np.empty((o_lat_size, o_lon_size), dtype=np.float32)
#     print('copy data')
#     filled_data = data.copy()

#     print('fill nanmask')
#     #
#     # Wherever `data` is NaN, set a 0 in `filled_data` and set a 1 in
#     # `nanmask`. `nanmask` will then be warped itself to find NaN values in the
#     # warped image.
#     #
#     for i_lat in range(o_lat_size):
#         for i_lon in range(o_lon_size):
#             if isnan(data[i_lat, i_lon]):
#                 nanmask[i_lat, i_lon] = 1
#                 filled_data[i_lat, i_lon] = 0
#                 nan_sum += 1
#             else:
#                 nanmask[i_lat, i_lon] = 0

#     print('map_coordinates filled_data')
#     result = map_coordinates(filled_data, coords,
#                              order=order, cval=cval)
#     lat_size = result.shape[0]
#     lon_size = result.shape[1]

#     print('map_coordinates nanmask')
#     if nan_sum > 0:
#         nanmask_mapped = map_coordinates(nanmask, coords, order=order, cval=1)
#         for i_lat in range(lat_size):
#             for i_lon in range(lon_size):
#                 if nanmask_mapped[i_lat, i_lon] > 0.9:
#                     result[i_lat, i_lon] = np.nan

#     return result


# cpdef affine_resample(dataset, extent, shape=None, resolution=None):
#     """Resample a dataset to a new extent.

#     Parameters
#     ----------
#     dataset : xarray.Dataset
#         The dataset to resample.
#     extent : array_like
#         The extent of the output dataset.
#     shape : tuple, optional
#         The shape of the output dataset.
#         Ignored if `resolution` is also passed.
#     resolution : tuple, optional
#         The resolution of the output dataset.

#     Returns
#     -------
#     xarray.Dataset
#         A new dataset with shape `shape` and extent `extent`.
#     """
#     cdef:
#         double lon_min, lat_min, lon_max, lat_max
#         double o_lon_min, o_lat_min, o_lon_max, o_lat_max
#         double lon_res, lat_res
#         double o_lon_res, o_lat_res
#         double lon_range, lat_range
#         double o_lon_range, o_lat_range
#         double lon_offset, lat_offset
#         np.ndarray[DOUBLE, ndim=1] new_coords_lat, new_coords_lon
#         SIZE_TYPE lat_size, lon_size
#         SIZE_TYPE o_lat_size, o_lon_size
#         SIZE_TYPE i_lat, i_lon

#     # Range and resolution of original dataset
#     o_extent = (dataset.lon.values.min(), dataset.lat.values.min(),
#                 dataset.lon.values.max(), dataset.lat.values.max())
#     o_lon_min, o_lat_min, o_lon_max, o_lat_max = o_extent
#     o_lat_size = dataset.sizes['lat']
#     o_lon_size = dataset.sizes['lon']
#     o_lat_range = o_lat_max - o_lat_min
#     o_lon_range = o_lon_max - o_lon_min
#     o_lat_res = o_lat_range / o_lat_size
#     o_lon_res = o_lon_range / o_lon_size

#     # Range and resolution of new dataset
#     lon_min, lat_min, lon_max, lat_max = extent
#     lat_range = lat_max - lat_min
#     lon_range = lon_max - lon_min

#     if resolution is not None:
#         lon_res, lat_res = resolution
#     elif shape is None:
#         raise ValueError("Need to pass either shape or resolution!")
#     else:
#         lat_size, lon_size = shape
#         lat_res = lat_range / lat_size
#         lon_res = lon_range / lon_size

#     # Offset
#     lat_offset = lat_min - o_lat_min
#     lon_offset = lon_min - o_lon_min
#     offset = [lat_offset, lon_offset]
#     # Affine transform
#     aff = np.array([lat_res / o_lat_res,
#                     lon_res / o_lon_res])
#     # Result dataset
#     new_coords = dict(dataset.coords)
#     new_coords_lat = np.linspace(lat_min, lat_max, lat_size)
#     new_coords_lon = np.linspace(lon_min, lon_max, lon_size)
#     new_coords['lat'] = new_coords_lat
#     new_coords['lon'] = new_coords_lon
#     result = xr.Dataset(coords=new_coords, attrs=dataset.attrs)
#     for var in dataset.data_vars:
#         result[var] = (('lat', 'lon'),
#                        affine_transform(dataset[var].values, aff,
#                                         offset=offset, order=1)
#                        )
#     return result