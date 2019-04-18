import xarray as xr
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn import preprocessing
except (ImportError, ModuleNotFoundError):
    raise ImportError('scikit-learn is required for this module.')

import numpy as np
import geopandas as gpd
import rasterio.features
from ..filters import BoxcarFilter
from ..warp import get_transform, nrows, ncols, get_bounds
from ..utils import get_vars_for_dims


__all__ = ['_cluster', '_cluster_smooth', 'cluster', 'norm_by_cluster']


def rasterize(shp, ds, columns=None):
    """Rasterize features to match a reference dataset.

    Parameters
    ----------
    shp : str or geopandas.geodataframe.GeoDataFrame
        Either the filename of a shapefile or an iterable
    ds : xarray.Dataset
        The reference dataset to match the raster shape.
    class_field : str, optional
        The name of field containing the label (default: 'class').
        Ignored if `shapes` is not a shapefile.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rasterized features.
    """
    ds_bbox = get_bounds(ds)
    if isinstance(shp, str):
        shp = gpd.read_file(shp, bbox=ds_bbox)
    layer = xr.Dataset(coords=ds.coords)
    shape = (len(layer['y']), len(layer['x']))
    for col, dtype in shp.dtypes.iteritems():
        if col in ['id', 'geometry']:
            continue
        layer[col] = (('y', 'x'), np.zeros(shape, dtype=dtype))

    for s in shp.iterfeatures():
        # mask is True outside the polygons.
        mask = rasterio.features.geometry_mask(
            [s['geometry']], shape, get_transform(ds)
        )
        for col, dtype in shp.dtypes.iteritems():
            if col not in layer.data_vars:
                if col in ['id', 'geometry']:
                    continue
                layer[col] = (('y', 'x'), np.zeros(shape, dtype=dtype))
            else:
                # Update the values inside the polygon to
                # the attribute of the polygon
                value = s['properties'][col]
                layer[col] = layer[col].where(mask, value)

    return layer


def _cluster(ds, ml=5, scale=True, variables=None, **kwargs):
    """
    Cluster a dataset using MiniBatchKMeans.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    ml : int, optional
        Boxcar window size for smoothing before clustering (default: 5).
    scale : bool, optional
        If True, scale the input data before clustering to zero mean and unit
        variance (default: True).
    variables : list of str, optional
        A list of variables to be used for clustering (default: all variables).
    kwargs : dict
        Extra keyword arguments passed on to MiniBatchKMeans.

    Returns
    -------
    clustered, labels : tuple (MiniBatchKMeans, xarray.DataArray)
        Returns the fitted MiniBatchKMeans instance and an xarray.DataArray
        with the cluster labels.
    """

    if variables is None:
        variables = [v for v in ds.data_vars
                     if 'y' in ds[v].coords and 'x' in ds[v].coords]
    # The redundant [variables] is necessary to remove the `time` coordinate
    df = ds[variables].to_dataframe()[variables]
    if ml == 0 or ml is None:
        values = df.values
    else:
        df_ml = BoxcarFilter(w=ml).apply(ds[variables]) \
            .to_dataframe()[variables]
        values = np.concatenate([df.values, df_ml.values], axis=1)
    if scale:
        values = preprocessing.scale(values)
    clust = MiniBatchKMeans(**kwargs).fit(values)
    df['label'] = clust.labels_
    return clust, df['label'].to_xarray()


def _cluster_smooth(ds, ml=5, scale=True, **kwargs):
    datavars = [v for v in ds.data_vars
                if 'y' in ds[v].coords and 'x' in ds[v].coords]
    # The redundant [datavars] is necessary to remove the `time` coordinate
    multilooked = BoxcarFilter(w=ml).apply(ds[datavars])
    # Calculate variance:
    diff = (ds - multilooked)
    diff_sq = diff * diff
    ds_var = BoxcarFilter(w=ml).apply(diff_sq[datavars]) * ml**2
    df_ml = multilooked.to_dataframe()[datavars]
    df_var = ds_var.to_dataframe()[datavars]
    values = np.concatenate([df_ml.values, df_var.values], axis=1)
    if scale:
        values = preprocessing.scale(values)
    clust = MiniBatchKMeans(**kwargs).fit(values)
    df_ml['label'] = clust.labels_
    return clust, df_ml['label'].to_xarray()


def cluster(ds, **kwargs):
    """
    At each time step, cluster the pixels according to their C2 matrix elements
    into `k` different clusters, using the previously found clusters as initial
    cluster centers for the next time step.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    kwargs : dict
        Extra keyword arguments to be passed on to _cluster().
    """
    ls = []
    km_args = kwargs.copy()
    for t in ds.time:
        clust, labels = _cluster(ds.sel(time=t), **km_args)
        km_args['init'] = clust.cluster_centers_
        km_args['n_init'] = 1
        ls.append(labels)
    cl = xr.concat(ls, 'time')
    cl['time'] = ds.time
    return cl


#
# Use the initial clustering result to normalize values.
#
def norm_by_cluster(ds, labels=None, n_clusters=10):
    """
    Norm each pixel by the average per cluster.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    labels : str, optional
        If None, cluster first (default: None).
    n_clusters : int, optional
        Only used if labels is None. Passed on to _cluster() (default: 10).
    """
    if labels is None:
        # Cluster the pixels at time 0.
        clust, labels = _cluster(ds.isel(time=0), n_clusters=n_clusters)

    # Assign to each pixel the mean of its cluster at time 0:
    _means = _clustermean(ds, labels)
    # _means0 = _clustermean(ds.isel(time=0), labels)
    _means0 = _means.isel(time=0)

    # Subtract the mean change within each cluster.
    normed = ds - _means + _means0

    # normed = ds.copy()
    # for l in range(n_clusters):
    #     _means = ds.where(labels == l).mean(dim=['lat', 'lon']).compute()
    #     _normed = ds - _means + _means0[l]
    #     # _normed = ds / _means * _means0
    #     normed = normed.where(labels != l).compute().fillna(_normed)

    return normed


def _clustermean(ds, labels):
    n_clusters = len(np.unique(labels))
    _means = ds.copy()
    for l in range(n_clusters):
        where = _means.where(labels == l)
        wherenot = _means.where(labels != l)
        _means = wherenot.fillna(
            where.mean(dim=['lat', 'lon']).compute()
        )
    return _means


def _build_X(ds):
    variables = get_vars_for_dims(ds, ('y', 'x'))
    data = ds[variables].to_array().transpose('y', 'x', 'variable').values
    return data.reshape((-1, len(variables)))


class Classifier:
    """
    Parameters
    ----------
    clf : sklearn classifier
        An initialized classifier object as provided by scikit-learn.
        Must provide methods ``fit`` and ``predict``.
    """

    def __init__(self, clf):
        self.clf = clf

    def fit(self, ds, labels):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset on which to train the classifier.
        labels : xarray.DataArray
            The class labels to train the classifier.
        """

        mask = np.array(np.logical_and(
            ~np.isnan(labels), labels > 0)).reshape(-1)
        X = _build_X(ds)[mask]
        y = np.array(labels).reshape(-1)[mask]
        self.clf.fit(X, y)

    def predict(self, ds, func='predict'):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset for which to predict the class labels.
        func : str, optional
            The method of the classifier to use for prediction
            (default: 'predict').

        Returns
        -------
        xarray.DataArray
            The predicted class labels.
        """

        if func not in dir(self.clf):
            raise AttributeError('Classifier has no method {}.'.format(func))
        X = _build_X(ds)
        labels_flat = self.clf.__getattribute__(func)(X)
        shape = (nrows(ds), ncols(ds))
        labels = xr.DataArray(labels_flat.reshape(shape), dims=('y', 'x'),
                              coords=ds.coords)
        return labels
