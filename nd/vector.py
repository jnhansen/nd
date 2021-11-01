"""A submodule to deal with vector data.
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio.features
import geopandas as gpd
import fiona
import shapely
import datetime
from . import warp


def read_file(path, clip=None):
    """
    Read a geospatial vector file.

    Parameters
    ----------
    path : str
        The path of the file to read.
    clip : shapely.geometry, optional
        A geometry to intersect the vector data.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if clip is None:
        return gpd.read_file(path)

    def records(filename, geom):
        with fiona.open(filename) as source:
            for feature in source:
                if geom is None:
                    yield feature
                else:
                    if feature['geometry'] is None:
                        continue
                    poly = shapely.geometry.shape(feature['geometry'])
                    if poly.intersects(geom):
                        yield feature

    return gpd.GeoDataFrame.from_features(records(path, clip))


def rasterize(shp, ds, columns=None, encode_labels=True, crs=None,
              date_field=None, date_fmt=None):
    """Rasterize a vector dataset to match a reference raster.

    Parameters
    ----------
    shp : str or geopandas.geodataframe.GeoDataFrame
        Either the filename of a shapefile or an iterable
    ds : xarray.Dataset
        The reference dataset to match the raster shape.
    columns : list of str, optional
        List of column names to read.
    encode_labels : bool, optional
        If True, convert categorical data to integer values. The corresponding
        labels are accessible in the metadata.
        (default: True).
    crs : str or dict or cartopy.crs.CRS, optional
        The CRS of the vector data.
    date_field : str, optional
        The name of field containing the timestamp.
    date_fmt : str, optional
        The date format to parse date_field. Passed to `pd.to_datetime()`.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rasterized features.
    """

    geom = warp.get_geometry(ds, crs=warp.get_crs(ds))
    transf = warp.get_transform(ds)

    # read and discard waste columns
    if isinstance(shp, str):
        shp = read_file(shp, clip=geom)
    else:
        # Work on a copy
        shp = shp.copy()

    if crs is not None:
        shp.crs = warp._parse_crs(crs)
    if shp.crs is not None:
        shp = shp.to_crs(warp.get_crs(ds))

    # Prepare output dataset
    layer = xr.Dataset(
        coords={
            'y': ds.coords['y'],
            'x': ds.coords['x']
        },
        attrs={
            'transform': tuple(transf[:6]),
            'crs': warp.get_crs(ds)
        })

    exclude_columns = ['geometry', date_field]

    # Convert date column to datetime format
    if date_field is None:
        # Create fake time coords
        # This is a bit of a hack:
        # I name the column None so that date_field still works
        shp[None] = pd.to_datetime(datetime.date.today())
        # coords = ('y', 'x')
        # shape = (len(layer['y']), len(layer['x']))

    else:
        if date_field not in shp:
            raise ValueError('Field {} does not exist.'.format(date_field))
        shp[date_field] = pd.to_datetime(shp[date_field], format=date_fmt)

    if columns is not None:
        # Avoid duplicate columns if 'geometry' or date_field have been
        # explicitly specified
        shp = shp[list(set(columns + ['geometry', date_field]))]

    # Add temporal coordinates
    layer.coords['time'] = np.unique(shp[date_field])
    coords = ('y', 'x', 'time')
    shape = (len(layer['y']), len(layer['x']), len(layer['time']))
    times = layer.coords['time'].values

    for c in shp.columns:
        if c in exclude_columns:
            continue

        data = shp[c]
        meta = {}

        # Treat object dtype as str
        if data.dtype in [object, str]:
            if encode_labels:
                # Encode categorical labels to integer
                data, legend = data.factorize()
                # Add 1 to reserve 0 for background
                data += 1
                # Create lookup table for storing in metadata
                meta['legend'] = list(enumerate([None] + list(legend)))

        # Prepare an empty DataArray
        layer[c] = (coords, np.empty(shape, dtype=data.dtype))

        for t in times:
            mask_t = (shp[date_field] == t)
            geom_t = shp[mask_t]['geometry']
            data_t = data[mask_t]

            if len(geom_t) == 0:
                continue

            if np.issubdtype(data.dtype, np.number):
                # Numeric dtype --> can rasterize directly.
                # Generate (geometry, value) pairs
                shp_data = list(zip(geom_t, data_t))

                # Rasterize
                layer[c].loc[dict(time=t)] = rasterio.features.rasterize(
                    shp_data,
                    out_shape=shape[:2],
                    transform=transf
                )

            else:
                # Create empty layer
                # layer[c] = (('y', 'x'), np.empty(shape, dtype=data.dtype))

                # Run through unique values
                unique_values = np.unique(
                    data_t[data_t.astype(bool)])
                for value in unique_values:
                    geoms = geom_t[data_t == value]
                    mask = rasterio.features.geometry_mask(
                        geoms, shape[:2], transf
                    )
                    layer[c].loc[dict(time=t)] = \
                        layer[c].loc[dict(time=t)].where(mask, value)

            layer[c].attrs = meta

    return layer
