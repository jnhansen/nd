"""
This module provides read/write capabilities for satellite data.
Internally all data is handled as xarray Datasets.

TODO: For any operation that affects nrows, ncols, or extent,
need to update the metadata!

"""
import lxml.etree as ET
from osgeo import gdal, osr, gdal_array
import os
# import glob
# import numpy as np
# from netCDF4 import Dataset
# import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.interpolation import map_coordinates
from . import utils
import affine
import re


try:
    type(profile)
except NameError:
    def profile(fn): return fn

NO_DATA_VALUE = np.nan


def _get_driver_from_filename(filename):
    if filename is None:
        return 'MEM'
    ext = os.path.splitext(filename)[1]
    if ext == '.tiff':
        return 'GTiff'
    else:
        return 'MEM'


def _make_gdal_dataset(data, src, extent=None, outfile=None, driver='auto'):
    """
    Create new GDAL dataset from given data.

    TODO: Implement. Right now this is the legacy code and it should be altered
    to accept an xarray Dataset.

    Parameters
    ----------
    data :
    src : gdal.Dataset
    driver : GDAL driver, optional
    outfile : optional
    """
    if not isinstance(data, list):
        data = [data]
    N_bands = len(data)
    N_lat, N_lon = data[0].shape
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data[0].dtype)
    if driver == 'auto':
        driver = _get_driver_from_filename(outfile)
    if driver == 'MEM':
        outfile = ''

    tmp = gdal.GetDriverByName(driver).Create(outfile, N_lon, N_lat, N_bands,
                                              gdal_dtype)

    for i in range(N_bands):
        new_band = tmp.GetRasterBand(i+1)
        new_band.WriteArray(data[i])
        new_band.SetNoDataValue(NO_DATA_VALUE)
        if src is not None:
            org_band = src.GetRasterBand(i+1)
            new_band.SetMetadata(org_band.GetMetadata())

    if extent is not None:
        # georeference
        output_lon_start = extent[0]
        output_lat_start = extent[1]
        lonlat_step = 0
        latlon_step = 0
        lat_step = (extent[3] - extent[1]) / (N_lat - 1)
        lon_step = (extent[2] - extent[0]) / (N_lon - 1)
        transform = (output_lon_start, lon_step, lonlat_step,
                     output_lat_start, latlon_step, lat_step)
        tmp.SetGeoTransform(transform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        tmp.SetProjection(srs.ExportToWkt())

    if src is not None:
        # Copy all metadata and projection information
        tmp.SetMetadata(src.GetMetadata())

    if driver != 'MEM':
        # write to disk
        tmp.FlushCache()
        tmp = None
        return outfile

    return tmp


@profile
def from_beam_dimap(path, read_data=True):
    """Read a BEAM Dimap product into an xarray Dataset.

    TODO: lat and lon should probably be single precision floats...

    Parameters
    ----------
    path : str
        The file path to the BEAM dimap product.
    read_data : bool, optional
        If True (default), read all data. Otherwise, read only the metadata.

    Returns
    -------
    xarray.Dataset
    """
    # -------------------------------------------------------------------------
    # Read metadata
    # -------------------------------------------------------------------------
    basepath = os.path.split(path)[0]
    meta = {}
    tree = ET.parse(path)
    root = tree.getroot()
    data_files = [os.path.join(basepath, _.attrib['href']) for _ in
                  root.findall('.//Data_File/DATA_FILE_PATH')]
    tie_point_grid_files = \
        [os.path.join(basepath, _.attrib['href']) for _ in
         root.findall('.//Tie_Point_Grid_File/TIE_POINT_GRID_FILE_PATH')]
    meta['ncols'] = int(root.find('.//Raster_Dimensions/NCOLS').text)
    meta['nrows'] = int(root.find('.//Raster_Dimensions/NROWS').text)
    meta['nbands'] = int(root.find('.//Raster_Dimensions/NBANDS').text)
    meta['time_start'] = root.find(
        './/Dataset_Sources//MDATTR[@name="first_line_time"]').text
    meta['orbit_direction'] = root.find(
        './/Dataset_Sources//MDATTR[@name="PASS"]').text
    meta['mode'] = root.find(
        './/Dataset_Sources//MDATTR[@name="ACQUISITION_MODE"]').text
    meta['rel_orbit'] = int(root.find(
        './/Dataset_Sources//MDATTR[@name="REL_ORBIT"]').text)
    meta['abs_orbit'] = int(root.find(
        './/Dataset_Sources//MDATTR[@name="ABS_ORBIT"]').text)
    meta['orbit_cycle'] = int(root.find(
        './/Dataset_Sources//MDATTR[@name="orbit_cycle"]').text)
    lats = [
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="first_near_lat"]').text),
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="first_far_lat"]').text),
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="last_near_lat"]').text),
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="last_far_lat"]').text)
    ]
    lons = [
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="first_near_long"]').text),
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="first_far_long"]').text),
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="last_near_long"]').text),
        float(root.find(
            './/Dataset_Sources//MDATTR[@name="last_far_long"]').text)
    ]
    meta['lon_range'] = (min(lons), max(lons))
    meta['lat_range'] = (min(lats), max(lats))

    # -------------------------------------------------------------------------
    # Determine the geolocation.
    # -------------------------------------------------------------------------

    # OPTION A:
    # Affine coordinate transform
    # ---------------------------
    crs_info = root.find('./Coordinate_Reference_System/WKT')
    transf_info = root.find('./Geoposition/IMAGE_TO_MODEL_TRANSFORM')

    # OPTION B:
    # Ground Control Points (Tie Point Grids)
    # ---------------------------------------
    tp_grids = {}
    for tf in tie_point_grid_files:
        p = os.path.splitext(tf)[0] + '.img'
        name = os.path.split(os.path.splitext(tf)[0])[1]
        tp_grids[name] = gdal.Open(p).ReadAsArray()

    #
    # Is an affine coord transform specified?
    #
    if crs_info is not None and transf_info is not None:
        #
        # Extract the GeoTransform.
        #
        transf = np.array([float(_) for _ in transf_info.text.split(',')])
        # The transform is in a weird order, reorder to GDAL standard:
        transf_gdal = transf[::-1].reshape((3, 2)).T.flatten()
        # Now reorder to affine transformation:
        aff = affine.Affine.from_gdal(*transf_gdal)
        # Store in metadata:
        meta['GeoTransform'] = transf_gdal

        #
        # Create coordinates
        #
        if aff.b == 0 and aff.d == 0:
            # The image is north-up, no need to store tie point grid.
            meta['pixel_height'] = np.abs(aff.a)
            meta['pixel_width'] = np.abs(aff.e)
            ys = np.stack([np.arange(meta['nrows']), np.zeros(meta['nrows'])])
            xs = np.stack([np.zeros(meta['ncols']), np.arange(meta['ncols'])])
            lat = (aff * ys)[0]
            lon = (aff * xs)[1]
            data_coords = ('lat', 'lon')
            coords = {'lat': lat, 'lon': lon}

        else:
            # The image is not north up.
            # Don't store additional coordinate data.
            # tpg_sparse = np.full((meta['nrows'], meta['ncols']), np.nan)
            data_coords = ('y', 'x')
            coords = {}

    #
    # Are there tie point grids for latitude and longitude?
    #
    elif 'latitude' in tp_grids and 'longitude' in tp_grids:
        # NOTE: In future, read these from the metadata.
        xstep = (meta['ncols'] - 1) / (tp_grids['latitude'].shape[1] - 1)
        ystep = (meta['nrows'] - 1) / (tp_grids['latitude'].shape[0] - 1)

        xs = np.linspace(0, meta['ncols'] - 1, tp_grids['latitude'].shape[1])
        ys = np.linspace(0, meta['nrows'] - 1, tp_grids['latitude'].shape[0])
        xi, yi = xs.astype(int), ys.astype(int)
        xg, yg = np.meshgrid(xi, yi, copy=False)
        # convert to integers so they can actually be assigned to pixels
        # don't want to carry over the error from the integer truncation,
        # therefore do an interpolation:
        # tpg_index = np.stack((yi, xi), axis=0)
        # xy_scaled = tpg_index.astype(float) / np.array((ystep, xstep))
        x_scaled = xg.astype(float) / xstep
        y_scaled = yg.astype(float) / ystep
        map_xy = np.stack((y_scaled, x_scaled), axis=0)
        tp_grids_int = {}
        for name, tpg in tp_grids.items():
            tp_grids_int[name] = map_coordinates(tpg, map_xy, output=tpg.dtype,
                                                 order=3, cval=np.nan)

        # Create GCP dataframe ...
        # gcp_data = {'GCPLine': y.flatten(), 'GCPPixel': x.flatten(),
        #             'GCPX': lon.flatten(), 'GCPY': lat.flatten()}
        # gcps = pd.DataFrame(gcp_data)

        # Create tie point grids as sparse matrix
        # (np.nan wherever there is no entry)
        tp_grids_sparse = {}
        # tpg_index = np.stack((y, x), axis=-1)
        for name, tpg in tp_grids_int.items():
            tpg_sparse = np.full((meta['nrows'], meta['ncols']), np.nan)
            tpg_sparse[yi[:, np.newaxis], xi] = tpg
            tp_grids_sparse[name] = tpg_sparse

        data_coords = ('y', 'x')
        coords = {'lat': (data_coords, tp_grids_sparse['latitude']),
                  'lon': (data_coords, tp_grids_sparse['longitude'])}

    #
    # Create xarray dataset.
    # `data_coords` and `coords` should have been defined at this point.
    #
    times = [utils.str2date(meta['time_start'])]
    coords['time'] = times
    ds = xr.Dataset(coords=coords, attrs=meta)

    if read_data:
        for rpath in data_files:
            # we don't want to open the ENVI .hdr file...
            im_path = os.path.splitext(rpath)[0] + '.img'
            name = os.path.splitext(os.path.split(im_path)[1])[0]
            band = gdal.Open(im_path)
            data = band.ReadAsArray()
            # ndv = band.GetNoDataValue()
            # data[data == ndv] = np.nan
            ds[name] = (data_coords, data)

    ds = _assemble_complex(ds)

    return ds


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
    write = _disassemble_complex(ds)
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
    ds = _assemble_complex(ds)
    return ds


def _disassemble_complex(ds):
    """Disassemble complex valued data into real and imag parts.

    Parameters
    ----------
    ds : xarray.Dataset
    """
    new_ds = ds.copy()
    #
    # Find all complex variables and disassemble into their real and
    # imaginary parts.
    #
    for vn, var in ds.data_vars.items():
        if not np.iscomplexobj(var):
            continue

        new_ds[vn + '__re'] = var.real
        new_ds[vn + '__im'] = var.imag
        del new_ds[vn]

    # reapply chunks
    new_ds = new_ds.chunk(ds.chunks)
    return new_ds


def _assemble_complex(ds, inplace=False):
    """Reassemble complex valued data.

    NOTE: Changes the dataset (view) in place!

    Parameters
    ----------
    ds : xarray.Dataset
    """
    if inplace:
        new_ds = ds
    else:
        new_ds = ds.copy()
    # find all variables that are meant to be complex
    endings = {'re': ['_real', '__re'],
               'im': ['_imag', '__im']}
    rex = {}
    matches = {}
    for part, end in endings.items():
        rex[part] = re.compile('(?P<stem>.*)(?:{})'.format('|'.join(end)))
        matches[part] = [rex[part].match(vn) for vn in ds.data_vars]
        matches[part] = [_ for _ in matches[part] if _ is not None]

    new_var_names = set([m.group('stem') for m in
                         matches['re'] + matches['im']])

    for vn in new_var_names:
        vn_re = utils.select(matches['re'], lambda x: x.group(1) == vn,
                             first=True)
        vn_im = utils.select(matches['im'], lambda x: x.group(1) == vn,
                             first=True)
        if vn_re is not None and vn_im is not None:
            new_ds[vn] = new_ds[vn_re.group(0)] + new_ds[vn_im.group(0)] * 1j
            del new_ds[vn_re.group(0)]
            del new_ds[vn_im.group(0)]

    # reapply chunks
    if not inplace:
        # new_ds = new_ds.chunk(ds.chunks)
        return new_ds


def _get_gcp_df(gdal_ds):
    """Converts the GCPs from the GDAL dataset into a pandas.DataFrame.

    Parameters
    ----------
    gdal_ds : osgeo.gdal.Dataset
        The dataset, including GCPs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns
        `['GCPLine', 'GCPPixel', 'GCPX', 'GCPY', 'GCPZ']`
    """
    gcps = gdal_ds.GetGCPs()
    attrs = ['GCPLine', 'GCPPixel', 'GCPX', 'GCPY', 'GCPZ']
    gcp_data = [{attr: getattr(gcp, attr) for attr in attrs} for gcp in gcps]
    return pd.DataFrame(gcp_data)


def from_gdal_dataset(gdal_ds):
    """Convert a GDAL dataset into an easier to handle xarray Dataset.

    Parameters
    ----------
    gdal_ds : gdal.Dataset or str
        Either an open GDAL dataset or a path that can be read by GDAL.

    Returns
    -------
    xarray.Dataset
    """
    if not isinstance(gdal_ds, gdal.Dataset):
        if isinstance(gdal_ds, str):
            gdal_ds = gdal.Open(gdal_ds)
        else:
            raise ValueError("`gdal_ds` is neither a GDAL dataset nor a valid "
                             "path to one.")
    ncols, nrows = gdal_ds.RasterXSize, gdal_ds.RasterYSize
    meta = gdal_ds.GetMetadata()
    times = [utils.str2date(meta['ACQUISITION_START_TIME'])]

    #
    # Determine whether the dataset uses GCPs or GeoTransform.
    #
    if gdal_ds.GetGCPCount() > 0:
        # Generate a sparse tie point grid of lat/lon coordinates
        # from the GCPs.
        gcps = _get_gcp_df(gdal_ds)
        llgrid = np.empty((nrows, ncols, 2))
        llgrid[:] = np.nan
        try:
            llgrid[gcps['GCPLine'].astype(int),
                   gcps['GCPPixel'].astype(int)] = gcps[['GCPY', 'GCPX']]
        except IndexError as e:
            raise IndexError("The specified dimensions of this dataset do not "
                             "match the GCPs (%s)." % e)
        lat = llgrid[:, :, 0]
        lon = llgrid[:, :, 1]
        result = xr.Dataset(coords={'lat': (['x', 'y'], lat),
                                    'lon': (['x', 'y'], lon),
                                    'time': times})
        data_coords = ('y', 'x')
    else:
        transform = gdal_ds.GetGeoTransform()
        lon_start, lon_step, _, lat_start, _, lat_step = transform
        lats = lat_start + np.arange(nrows) * lat_step
        lons = lon_start + np.arange(ncols) * lon_step
        result = xr.Dataset(coords={'lat': lats, 'lon': lons, 'time': times},
                            attrs=meta)
        data_coords = ('lat', 'lon')

    # write each raster individually.
    for i in range(gdal_ds.RasterCount):
        band = gdal_ds.GetRasterBand(i+1)
        ndv = band.GetNoDataValue()
        name = band.GetMetadata()['POLARISATION']
        data = band.ReadAsArray()
        data[data == ndv] = np.nan
        result[name] = (data_coords, data)

    return result


if __name__ == '__main__':
    pass
