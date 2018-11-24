from .. import utils
from osgeo import gdal, osr, gdal_array
import os
import numpy as np
import pandas as pd
import xarray as xr


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
    if 'ACQUISITION_START_TIME' in meta:
        times = [utils.str2date(meta['ACQUISITION_START_TIME'])]
    else:
        times = None

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
        result = xr.Dataset(coords={'lat': (['y', 'x'], lat),
                                    'lon': (['y', 'x'], lon),
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
        bandmeta = band.GetMetadata()
        if 'POLARISATION' in bandmeta:
            name = bandmeta['POLARISATION']
        else:
            name = '{}'.format(i)
        data = band.ReadAsArray()
        nanmask = (data == ndv)
        if nanmask.any():
            data = data.astype(np.float)
            data[nanmask] = np.nan
        result[name] = (data_coords, data)

    return result
