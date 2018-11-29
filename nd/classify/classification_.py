import xarray as xr
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn import preprocessing
except (ImportError, ModuleNotFoundError):
    raise ImportError('scikit-learn is required for this module.')

import numpy as np
from ..filters import boxcar


__all__ = ['_cluster', '_cluster_smooth', 'cluster', 'norm_by_cluster']


def _cluster(ds, ml=5, scale=True, variables=None, **kwargs):
    if variables is None:
        variables = [v for v in ds.data_vars
                     if 'lat' in ds[v].coords and 'lon' in ds[v].coords]
    # The redundant [variables] is necessary to remove the `time` coordinate
    df = ds[variables].to_dataframe()[variables]
    if ml == 0 or ml is None:
        values = df.values
    else:
        df_ml = boxcar(ds[variables], w=ml).to_dataframe()[variables]
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
    multilooked = boxcar(ds[datavars], w=ml)
    # Calculate variance:
    diff = (ds - multilooked)
    diff_sq = diff * diff
    ds_var = boxcar(diff_sq[datavars], w=ml) * ml**2
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
