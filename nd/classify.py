try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn import preprocessing
except (ImportError, ModuleNotFoundError):
    raise ImportError('scikit-learn is required for this module.')

import xarray as xr
import numpy as np
from .filters import BoxcarFilter
from . import utils
from collections import OrderedDict


__all__ = ['Classifier', '_cluster', '_cluster_smooth', 'cluster',
           'norm_by_cluster']


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


def _build_X(ds, feature_dims=[]):
    data_dims = _get_data_dims(ds, feature_dims=feature_dims)
    variables = utils.get_vars_for_dims(ds, data_dims)
    features = tuple(feature_dims) + ('variable',)
    data = ds[variables].to_array().stack(feature=features).transpose(
        *data_dims, 'feature', transpose_coords=True).values
    return data.reshape((-1, data.shape[-1]))


def _get_data_dims(ds, feature_dims=[]):
    data_dims = tuple([d for d in ds.coords if d in ds.dims
                       and d not in feature_dims])
    return data_dims


def _get_data_shape(ds, feature_dims=[]):
    data_dims = _get_data_dims(ds, feature_dims=feature_dims)
    shape = tuple([ds.sizes[d] for d in data_dims])
    return shape


def _broadcast_array(arr, shape):
    matching = list(shape)
    new_shape = [1] * len(shape)
    for dim in arr.shape:
        i = matching.index(dim)
        new_shape[i] = dim
        matching[i] = None
    return np.broadcast_to(arr.reshape(new_shape), shape)


def _broadcast_labels(labels, ds, feature_dims=[]):
    shape = _get_data_shape(ds, feature_dims=feature_dims)
    if isinstance(labels, np.ndarray):
        # Broadcast for np.ndarray
        return _broadcast_array(labels, shape)

    elif isinstance(labels, xr.DataArray):
        # Broadcast for xarray.DataArray
        data_dims = _get_data_dims(ds, feature_dims=feature_dims)
        # Determine dimensions to be broadast:
        bc_dims = set(data_dims) - set(labels.dims) - \
            set(feature_dims)
        for dim in bc_dims:
            labels = xr.concat([labels] * ds.sizes[dim], dim=dim)
            labels.coords[dim] = ds.coords[dim]
        labels = labels.transpose(*data_dims)
        return labels


class Classifier:
    """
    Parameters
    ----------
    clf : sklearn classifier
        An initialized classifier object as provided by scikit-learn.
        Must provide methods ``fit`` and ``predict``.
    feature_dims : list, optional
        A list of additional dimensions to use as features.
        For example, if the dataset has a 'time' dimension and 'time' is in
        ``feature_dims``, every time step will be treated as an independent
        variable for classification purposes.
        Otherwise, all time steps will be treated as additional data
        dimensions just like 'x' and 'y'.

    """

    def __init__(self, clf, feature_dims=[]):
        self.clf = clf
        self.feature_dims = feature_dims

    def fit(self, ds, labels):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset on which to train the classifier.
        labels : xarray.DataArray
            The class labels to train the classifier.
        """
        if isinstance(labels, xr.Dataset):
            raise ValueError("`labels` should be an xarray.DataArray or "
                             "numpy array of the same dimensions "
                             "as the dataset.")
        elif isinstance(labels, (xr.DataArray, np.ndarray)):
            # Squeeze extra dimensions
            labels = labels.squeeze()

        # Broadcast labels to match data dimensions
        shape = _get_data_shape(ds, feature_dims=self.feature_dims)
        if shape != labels.shape:
            labels = _broadcast_labels(
                labels, ds, feature_dims=self.feature_dims)

        # Ignore labels that are NaN or 0
        ymask = ~np.isnan(np.array(labels))
        np.greater(labels, 0, out=ymask, where=ymask)
        ymask = ymask.reshape(-1)

        # Ignore values of ds that contain NaN
        X = _build_X(ds, feature_dims=self.feature_dims)[ymask]
        y = np.array(labels).reshape(-1)[ymask]
        Xmask = ~np.isnan(X).any(axis=1)
        self.clf.fit(X[Xmask], y[Xmask])
        return self

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
        X = _build_X(ds, feature_dims=self.feature_dims)
        # Skip prediction for NaN entries:
        mask = ~np.isnan(X).any(axis=1)
        labels_flat = np.empty(mask.shape) * np.nan
        labels_flat[mask] = self.clf.__getattribute__(func)(X[mask])
        data_dims = _get_data_dims(ds, feature_dims=self.feature_dims)
        data_shape = _get_data_shape(ds, feature_dims=self.feature_dims)
        data_coords = OrderedDict(
            (dim, c) for dim, c in ds.coords.items() if dim in data_dims
        )
        labels = xr.DataArray(labels_flat.reshape(data_shape),
                              dims=data_dims, coords=data_coords)
        return labels

    def fit_predict(self, ds, labels):
        self.fit(ds, labels)
        return self.predict(ds)
