import pytest
from nd.testing import (generate_test_dataset, generate_test_geodataframe,
                        assert_equal_crs)
from nd import vector
from nd import warp
from numpy.testing import assert_equal, assert_allclose
from geopandas.testing import assert_geodataframe_equal
import geopandas as gpd
import numpy as np
import rasterio
from scipy import ndimage


def test_rasterize_no_side_effects():
    ds = generate_test_dataset()
    df = generate_test_geodataframe()
    df_copy = df.copy()
    _ = vector.rasterize(df, ds)
    # Check that the original GeoDataFrame doesn't change as part of the
    # rasterization
    assert_geodataframe_equal(
        df, df_copy
    )


def test_rasterize(tmpdir):
    path = str(tmpdir.join('polygons.shp'))
    ds = generate_test_dataset(dims=dict(x=100, y=100, time=5))
    df = generate_test_geodataframe()
    schema = gpd.io.file.infer_schema(df)
    schema['properties']['date'] = 'date'
    df.to_file(path, schema=schema)

    # Rasterize
    raster = vector.rasterize(path, ds)

    # Check that the raster contains all fields as variables
    assert set(raster.data_vars).union({'geometry'}) == set(df.columns)

    # Check dtypes
    assert np.issubdtype(raster.float.dtype, np.floating)
    assert np.issubdtype(raster.integer.dtype, np.signedinteger)
    assert np.issubdtype(raster.category.dtype, np.signedinteger)

    # Check that extent, projection etc. are identical to the reference raster
    assert_equal(
        warp.get_bounds(raster),
        warp.get_bounds(ds)
    )
    assert_equal_crs(
        warp.get_crs(raster),
        warp.get_crs(ds)
    )
    assert_equal(
        warp.get_transform(raster),
        warp.get_transform(ds)
    )

    # Check raster content
    shape = (ds.dims['y'], ds.dims['x'])
    transform = warp.get_transform(ds)
    for i, row in df.iterrows():
        poly = row['geometry']
        mask = rasterio.features.rasterize(
            [poly], out_shape=shape, transform=transform
        )
        # Erode mask to avoid edge effects
        mask = ndimage.morphology.binary_erosion(mask) == 1

        for v in raster.data_vars:
            if 'legend' in raster[v].attrs:
                expected = sorted(raster[v].attrs['legend'],
                                  key=lambda x: x[1] == str(row[v]))[-1][0]
            else:
                expected = row[v]
            values = raster[v].isel(time=0).values
            values[mask]
            assert_allclose(values[mask], expected)


@pytest.mark.parametrize('columns', [
    ['integer'],
    ['integer', 'date'],
    ['float', 'category'],
    ['integer', 'geometry'],
])
@pytest.mark.parametrize('date_field', ['date', None])
def test_rasterize_columns(columns, date_field):
    ds = generate_test_dataset()
    df = generate_test_geodataframe()
    raster = vector.rasterize(df, ds, columns=columns,
                              date_field=date_field)
    if date_field is None:
        expected_vars = set(columns) - {'geometry'}
    else:
        expected_vars = set(columns) - {'geometry', 'date'}

    assert_equal(
        set(raster.data_vars),
        expected_vars
    )


def test_rasterize_date_field():
    ds = generate_test_dataset()
    df = generate_test_geodataframe()
    raster = vector.rasterize(df, ds, date_field='date')

    assert len(np.unique(df['date'])) == raster.dims['time']

    assert_equal(
        np.unique(df['date']).astype('datetime64[s]'),
        raster.time.values.astype('datetime64[s]')
    )
