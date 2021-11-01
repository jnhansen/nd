try:
    from sklearn import preprocessing, metrics
except (ImportError, ModuleNotFoundError):
    raise ImportError('scikit-learn is required for this module.')

import xarray as xr
import numpy as np
from . import utils
from collections import OrderedDict


__all__ = ['Classifier', 'class_mean']


def class_mean(ds, labels):
    """
    Replace every pixel of the dataset with the mean of its corresponding
    class (or cluster, or segment).

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset.
    labels : xarray.DataArray
        The labels indicating the class each pixel in ``ds`` belongs to.
        The label dimensions may be a subset of the dimensions of ``ds``
        (such as ``('y', 'x')`` for a dataset that also contains a ``time``
        dimension but with time-independent class labels).

    Returns
    -------
    xarray.Dataset
        A dataset with the corresponding mean class values.
    """

    n_clusters = len(np.unique(labels))
    _means = ds.copy()
    for l in range(n_clusters):
        where = _means.where(labels == l)
        wherenot = _means.where(labels != l)
        _means = wherenot.fillna(
            where.mean().compute()
        )
    return _means


def _build_X(ds, feature_dims=[]):
    data_dims = _get_data_dims(ds, feature_dims=feature_dims)
    features = tuple(feature_dims) + ('variable',)

    if isinstance(ds, xr.Dataset):
        variables = utils.get_vars_for_dims(ds, data_dims)
        data = ds[variables].to_array()
    else:
        data = ds.expand_dims('variable')

    data = data.stack(feature=features).transpose(
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
        labels = labels.transpose(*data_dims, transpose_coords=True)
        return labels


class Classifier:
    """
    Parameters
    ----------
    clf : sklearn classifier
        An initialized classifier object as provided by ``scikit-learn``.
        Must provide methods ``fit`` and ``predict``.
    feature_dims : list, optional
        A list of additional dimensions to use as features.
        For example, if the dataset has a ``'time'`` dimension and ``'time'``
        is in ``feature_dims``, every time step will be treated as an
        independent variable for classification purposes.
        Otherwise, all time steps will be treated as additional data
        dimensions just like ``'x'`` and ``'y'``.
    scale : bool, optional
        If True, scale the input data before clustering to zero mean and unit
        variance (default: False).

    """

    def __init__(self, clf, feature_dims=[], scale=False):
        self.clf = clf
        self.feature_dims = feature_dims
        self.scale = scale
        self._scaler = None

    def make_Xy(self, ds, labels=None):
        """
        Generate scikit-learn compatible X and y arrays
        from `ds` and `labels`.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset.
        labels : xarray.DataArray
            The corresponding class labels.

        Returns
        -------
        tuple(np.array, np.array)
            X and y
        """

        if isinstance(labels, xr.Dataset):
            raise ValueError("`labels` should be an xarray.DataArray or "
                             "numpy array of the same dimensions "
                             "as the dataset.")
        elif isinstance(labels, (xr.DataArray, np.ndarray)):
            # Squeeze extra dimensions
            labels = labels.squeeze()

        if labels is not None:
            # Broadcast labels to match data dimensions
            # shape = _get_data_shape(ds, feature_dims=self.feature_dims)
            # The shape check may sometimes incorrectly pass
            # if e.g. x and y have the same size
            # if shape != labels.shape:
            labels = _broadcast_labels(
                labels, ds, feature_dims=self.feature_dims)

            # Ignore labels that are NaN or 0
            ymask = ~np.isnan(np.array(labels))
            np.greater(labels, 0, out=ymask, where=ymask)
            ymask = ymask.reshape(-1)
        else:
            # Unsupervised methods don't take a labels argument
            ymask = slice(None)

        # Ignore values of ds that contain NaN
        X = _build_X(ds, feature_dims=self.feature_dims)[ymask]
        Xmask = ~np.isnan(X).any(axis=1)
        X = X[Xmask]

        if labels is not None:
            y = np.array(labels).reshape(-1)[ymask][Xmask]
        else:
            y = None

        if self.scale:
            self._scaler = preprocessing.StandardScaler()
            self._scaler.fit(X)
            X = self._scaler.transform(X)

        return (X, y)

    def fit(self, ds, labels=None):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset on which to train the classifier.
        labels : xarray.DataArray, optional
            The class labels to train the classifier. To be omitted
            if the classifier is unsupervised, such as KMeans.
        """

        X, y = self.make_Xy(ds, labels=labels)
        self.clf.fit(X, y)
        return self

    def predict(self, ds, func='predict'):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset for which to predict the class labels.
        func : str, optional
            The method of the classifier to use for prediction
            (default: ``'predict'``).

        Returns
        -------
        xarray.DataArray
            The predicted class labels.
        """

        if func not in dir(self.clf):
            raise AttributeError('Classifier has no method {}.'.format(func))
        X = _build_X(ds, feature_dims=self.feature_dims)
        # Skip prediction for NaN entries, but
        # maintain original shape for labels
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        # Scale according to fit
        if self.scale:
            X = self._scaler.transform(X)

        result = self.clf.__getattribute__(func)(X)
        data_dims = _get_data_dims(ds, feature_dims=self.feature_dims)
        data_shape = _get_data_shape(ds, feature_dims=self.feature_dims)
        data_coords = OrderedDict(
            (dim, c) for dim, c in ds.coords.items() if dim in data_dims
        )

        labels_flat = np.empty(mask.shape + result.shape[1:]) * np.nan
        labels_flat[mask] = result
        labels_data = labels_flat.reshape(data_shape + result.shape[1:])
        if len(result.shape) > 1:
            # e.g. when func == 'predict_proba'
            data_dims = data_dims + ('label',)
            data_shape = data_shape + result.shape[1:]
            data_coords['label'] = np.arange(result.shape[1])

        labels = xr.DataArray(labels_data,
                              dims=data_dims, coords=data_coords)
        return labels

    def fit_predict(self, ds, labels=None):
        self.fit(ds, labels)
        return self.predict(ds)

    def score(self, ds, labels=None, method='accuracy'):
        """
        Compute the classification score.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset for which to compute the score.
        labels : xarray.DataArray
            The corresponding true class labels.
        method : str, optional
            The scoring method as implemented in scikit-learn
            (default: 'accuracy')

        Returns
        -------
        float
            The classification score.
        """

        try:
            scorer = metrics.get_scorer(method)
        except Exception:
            raise ValueError(
                "'{}' is not a valid scoring method".format(method))

        X, y = self.make_Xy(ds, labels=labels)

        return scorer(self.clf, X, y)
