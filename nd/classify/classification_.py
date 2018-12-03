import xarray as xr
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn import preprocessing
except (ImportError, ModuleNotFoundError):
    raise ImportError('scikit-learn is required for this module.')

import numpy as np
from ..filters import BoxcarFilter


__all__ = ['_cluster', '_cluster_smooth', 'cluster', 'norm_by_cluster']


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
                     if 'lat' in ds[v].coords and 'lon' in ds[v].coords]
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
                if 'lat' in ds[v].coords and 'lon' in ds[v].coords]
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


class Classifier:

    def __init__(self, cls):
        self.cls = cls

    def fit(self, ds, labels):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset on which to train the classifier.
        labels : xarray.DataArray
            The class labels to train the classifier.
        """
        self.cls.fit(...)
        pass

    def predict(self, ds):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset for which to predict the class labels.

        Returns
        -------
        xarray.DataArray
            The predicted class labels.
        """
        # labels = self.cls.predict(...)
        pass
