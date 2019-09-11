"""A submodule to deal with vector data.
"""

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import LabelEncoder
import rasterio.features
import geopandas as gpd
import datetime
from . import warp


def rasterize(shp, ds, columns=None, encode_labels=True, date_field=None):
    """Rasterize a vector dataset to match a reference raster.

    Parameters
    ----------
    shp : str or geopandas.geodataframe.GeoDataFrame
        Either the filename of a shapefile or an iterable
    ds : xarray.Dataset
        The reference dataset to match the raster shape.
    encode_labels : bool, optional
        If True, convert categorical data to integer values. The corresponding
        labels are accessible in the metadata.
        (default: True).
    date_field : str, optional
        The name of field containing the timestamp.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rasterized features.
    """

    bbox = warp.get_bounds(ds)
    transf = warp.get_transform(ds)

    # read and discard waste columns
    if isinstance(shp, str):
        shp = gpd.read_file(shp, bbox=bbox)
    else:
        # Work on a copy
        shp = shp.copy()

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
        shp[date_field] = pd.to_datetime(shp[date_field])

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
        if data.dtype in [np.object, np.str]:
            if encode_labels:
                # Encode categorical labels to integer
                # Add 1 to reserve 0 for background
                enc = LabelEncoder()
                enc.fit(data.astype(str))
                data = enc.transform(data.astype(str)) + 1
                # Create lookup table for storing in metadata
                meta['legend'] = list(enumerate(
                    np.insert(enc.classes_, 0, None)))

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
