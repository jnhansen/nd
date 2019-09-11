from . import utils
import lxml.etree as ET
import rasterio as rio
import numpy as np
import os
import re
import affine
import xarray as xr
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates

__all__ = ['open_dataset',
           'open_netcdf',
           'open_beam_dimap',
           'open_rasterio',
           'to_netcdf',
           'assemble_complex',
           'disassemble_complex',
           'add_time']


# --------------------
# CONVERSION FUNCTIONS
# --------------------

def disassemble_complex(ds, inplace=False):
    """Disassemble complex valued data into real and imag parts.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with complex variables.
    inplace : bool, optional
        Whether to modify the dataset inplace (default: False).

    Returns
    -------
    xarray.Dataset or None
        If inplace, returns None. Otherwise, returns a dataset where all
        complex variables have been split into their real and imaginary parts.
    """

    if isinstance(ds, xr.DataArray):
        name = ds.name
        if name is None:
            name = 'data'
        ds = ds.to_dataset(name=name)

    if inplace:
        new_ds = ds
    else:
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
    if len(ds.chunks) > 0:
        new_ds = new_ds.chunk(ds.chunks)
    if not inplace:
        return new_ds


def assemble_complex(ds, inplace=False):
    """Reassemble complex valued data.

    NOTE: Changes the dataset (view) in place!

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with complex variables split into real and imaginary
        parts.
    inplace : bool, optional
        Whether to modify the dataset inplace (default: False).

    Returns
    -------
    xarray.Dataset or None
        If inplace, returns None. Otherwise, returns a dataset where the
        real and imaginary parts have been combined into the respective
        complex variables.
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


def add_time(ds, inplace=False):
    """
    Add a `time` dimension to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.

    Returns
    -------
    xarray.Dataset
        A dataset that is guaranteed to contain the dimension `time`.
    """
    if not inplace:
        result = ds.copy()
    else:
        result = ds
    if 'time' not in result.coords:
        times = [utils.str2date(ds.attrs['start_date'])]
        result.coords['time'] = times
    if not inplace:
        return result


# -------------
# OPEN DATASETS
# -------------

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


# --------------
# FORMAT: NETCDF
# --------------

def to_netcdf(ds, path, *args, **kwargs):
    """Write an xarray Dataset to disk.

    In addition to ``xarray.to_netcdf``, this function allows to store complex
    valued data by converting it to a a pair of reals. This process is
    reverted when reading the file via ``from_netcdf``.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be stored to disk.
    path : str
        The path of the target NetCDF file.
    *args : list
        Extra positional arguments for ``xr.Dataset.to_netcdf``.
    **kwargs : dict
        Extra keyword arguments for ``xr.Dataset.to_netcdf``.
    """

    write = disassemble_complex(ds)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in write.variables}
    if 'encoding' in kwargs:
        encoding.update(kwargs['encoding'])
    kwargs['encoding'] = encoding
    if 'engine' not in kwargs:
        kwargs['engine'] = 'h5netcdf'
    return write.to_netcdf(path, *args, **kwargs)


def open_netcdf(path, as_complex=False, *args, **kwargs):
    """Read a NetCDF file into an xarray Dataset.

    Wrapper function for ``xarray.open_dataset`` that preserves complex
    valued data.

    Parameters
    ----------
    path : str
        The path of the NetCDF file to read.
    as_complex : bool, optional
        Whether or not to assemble real and imaginary parts into complex
        (default: False).
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

    if 'engine' not in kwargs:
        kwargs['engine'] = 'h5netcdf'
    ds = xr.open_dataset(path, *args, **kwargs)
    if as_complex:
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


# ---------------------
# FORMAT: GDAL READABLE
# ---------------------

def open_rasterio(path, *args, **kwargs):
    return xr.open_rasterio(path, *args, **kwargs)


# ------------------
# FORMAT: BEAM DIMAP
# ------------------

def open_beam_dimap(path, read_data=True, as_complex=True):
    """Read a BEAM Dimap product into an xarray Dataset.

    BEAM Dimap is the native file format of the SNAP software. It consists of
    a ``*.dim`` XML file and a ``*.data`` directory containing the data.
    ``path`` should point to the XML file.

    Parameters
    ----------
    path : str
        The file path to the BEAM Dimap product.
    read_data : bool, optional
        If True (default), read all data. Otherwise, read only the metadata.

    Returns
    -------
    xarray.Dataset
        The same dataset converted into xarray.
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
        with rio.open(p) as src:
            # Read first bands
            tp_grids[name] = src.read(1)

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
            ds[name] = xr.open_rasterio(im_path)

        # All attributes that are the same for each band
        # should be attributes of the dataset instead.
        attrs = pd.DataFrame(ds[v].attrs for v in ds.data_vars)
        for col in attrs.columns:
            if len(attrs[col].unique()) == 1:
                ds.attrs[col] = attrs[col][0]
                for v in ds.data_vars:
                    del ds[v].attrs[col]

    if as_complex:
        ds = assemble_complex(ds)

    return ds
