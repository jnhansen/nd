from .. import utils
from .convert_ import assemble_complex
import lxml.etree as ET
import rasterio as rio
import numpy as np
import os
import affine
import xarray as xr
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates


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
