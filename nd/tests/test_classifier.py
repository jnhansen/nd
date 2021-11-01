import pytest
import numpy as np
from nd import utils
from nd import classify
from nd.testing import create_mock_classes
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import find_boundaries
from sklearn.exceptions import NotFittedError
from sklearn.cluster import MiniBatchKMeans
from numpy.testing import assert_equal, assert_raises_regex, assert_allclose
from xarray.testing import assert_equal as xr_assert_equal
from collections import OrderedDict


@pytest.mark.parametrize('clf', [
    GaussianNB(),
    KNeighborsClassifier(3),
    RandomForestClassifier(n_estimators=20),
])
@pytest.mark.parametrize('array', (True, False))
def test_classifier(clf, array):
    dims = OrderedDict([('y', 50), ('x', 50)])
    ds, labels_true = create_mock_classes(dims)
    if array:
        ds = ds['C11']

    # Select 10% for training
    labels_train = labels_true.copy()
    mask_train = (np.random.rand(dims['y'], dims['x']) < 0.1)
    labels_train = labels_train.where(mask_train)
    c = classify.Classifier(clf)
    c.fit(ds, labels_train)
    labels_predicted = c.predict(ds)

    # Expect 100% accuracy for this trivial classification task.
    xr_assert_equal(labels_predicted, labels_true)


@pytest.mark.parametrize('dims', [
    OrderedDict([('y', 50), ('x', 50), ('time', 10)]),
    OrderedDict([('y', 30), ('x', 20), ('time', 5)])
])
@pytest.mark.parametrize('feature_dims', [
    [], ['time']
])
def test_broadcast(dims, feature_dims):
    ds, labels = create_mock_classes(dims)

    expected_shape = classify._get_data_shape(ds, feature_dims=feature_dims)

    # Check broadcast for numpy array
    blabels = classify._broadcast_labels(labels.values, ds, feature_dims)
    assert blabels.shape == expected_shape

    # Check broadcast for DataArray
    blabels = classify._broadcast_labels(
        labels.transpose('x', 'y'), ds, feature_dims)
    blabels_t = classify._broadcast_labels(
        labels.transpose('y', 'x'), ds, feature_dims)
    xr_assert_equal(blabels, blabels_t)
    assert blabels.shape == expected_shape

    # Check values equal along broadcast dimensions
    bc_dims = set(dims) - set(labels.dims) - set(feature_dims)
    for d in bc_dims:
        assert (blabels.std(d) == 0).all()


@pytest.mark.parametrize('feature_dims', [
    [], ['time']
])
def test_build_X(feature_dims):
    dims = OrderedDict([('y', 50), ('x', 50), ('time', 10)])
    ds, labels = create_mock_classes(dims)
    X = classify._build_X(ds, feature_dims=feature_dims)
    nrows = np.prod([N for d, N in dims.items() if d not in feature_dims])
    ncols = len(ds.data_vars) * \
        np.prod([N for d, N in dims.items() if d in feature_dims])
    assert X.shape == (nrows, ncols)

    # Test that the original dimension order doesn't matter
    ds_t = ds.transpose(*list(reversed(dims)))
    X_t = classify._build_X(ds_t, feature_dims=feature_dims)
    assert_equal(X, X_t)


@pytest.mark.parametrize('feature_dims', [
    [], ['time']
])
@pytest.mark.parametrize('dims', [
    OrderedDict([('y', 50), ('x', 50), ('time', 10)]),
    OrderedDict([('y', 30), ('x', 20), ('time', 5)])
])
def test_classifier_feature_dims(dims, feature_dims):
    ds, labels = create_mock_classes(dims)
    c = classify.Classifier(RandomForestClassifier(n_estimators=20),
                            feature_dims=feature_dims)

    # Expect 100% accuracy for this trivial classification task.
    pred = c.fit(ds, labels).predict(ds)
    xr_assert_equal(
        pred, classify._broadcast_labels(labels, ds, feature_dims=feature_dims)
    )

    # Check that the results are the same whether labels
    # are passed as xr.DataArray or np.ndarray
    pred_np = c.fit(ds, labels.values).predict(ds)
    xr_assert_equal(pred, pred_np)

    # Check that prediction result has correct dimensions
    assert_equal(
        utils.get_dims(pred),
        classify._get_data_dims(ds, feature_dims=feature_dims)
    )


def test_fit_predict():
    dims = OrderedDict([('y', 50), ('x', 50), ('time', 10)])
    ds, labels = create_mock_classes(dims)
    c = classify.Classifier(RandomForestClassifier(n_estimators=20))
    xr_assert_equal(
        c.fit(ds, labels).predict(ds),
        c.fit_predict(ds, labels)
    )


def test_predict_proba():
    dims = OrderedDict([('y', 50), ('x', 50), ('time', 10)])
    ds, labels = create_mock_classes(dims)
    c = classify.Classifier(LogisticRegression())
    pred = c.fit(ds, labels).predict(ds)
    proba = c.fit(ds, labels).predict(ds, func='predict_proba')
    assert (proba >= 0).all()
    assert (proba <= 1).all()
    xr_assert_equal(
        pred,
        proba.argmax('label') + 1
    )


def test_predict_before_fit():
    dims = OrderedDict([('y', 100), ('x', 100)])
    ds, true_labels = create_mock_classes(dims)
    c = classify.Classifier(RandomForestClassifier())
    with assert_raises_regex(NotFittedError, 'not fitted yet'):
        c.predict(ds)


def test_score():
    dims = OrderedDict([('y', 100), ('x', 100)])
    ds, labels = create_mock_classes(dims)
    c = classify.Classifier(RandomForestClassifier())
    pred = c.fit(ds, labels).predict(ds)
    true_score = (pred == labels).mean()
    test_score = c.score(ds, labels)
    assert true_score == test_score


def test_score_invalid_method():
    dims = OrderedDict([('y', 100), ('x', 100)])
    ds, labels = create_mock_classes(dims)
    c = classify.Classifier(RandomForestClassifier())
    c.fit(ds, labels)
    with assert_raises_regex(ValueError, 'not a valid scoring method'):
        c.score(ds, labels, method='not_a_method')


def test_scaling():
    dims = OrderedDict([('y', 100), ('x', 100)])
    ds, true_labels = create_mock_classes(dims)

    # Check that scaling doesn't affect the prediction
    # for a simple classification task

    clf_noscale = classify.Classifier(
        MiniBatchKMeans(n_clusters=2), scale=False).fit(ds)
    clf_scale = classify.Classifier(
        MiniBatchKMeans(n_clusters=2), scale=True).fit(ds)

    assert_equal(
        find_boundaries(clf_noscale.predict(ds)),
        find_boundaries(clf_scale.predict(ds))
    )

    # Check that scaler has been properly fitted
    assert_allclose(
        clf_scale._scaler.mean_,
        ds.mean().to_array().values
    )
    assert_allclose(
        clf_scale._scaler.scale_,
        ds.std().to_array().values
    )

    # Scaled cluster center mean should be about zero
    assert np.all(
        np.abs(clf_scale.clf.cluster_centers_.mean(axis=0)) < 1e-2
    )


# ----------
# Clustering
# ----------

def test_cluster():
    dims = OrderedDict([('y', 100), ('x', 100)])
    ds, true_labels = create_mock_classes(dims)
    clf = classify.Classifier(MiniBatchKMeans(n_clusters=2))
    clustered = clf.fit_predict(ds)

    # Check that the clusters are identical to the true labels
    assert_equal(
        find_boundaries(true_labels),
        find_boundaries(clustered)
    )


def test_class_mean():
    dims = OrderedDict([('y', 100), ('x', 100)])
    ds, true_labels = create_mock_classes(dims)
    means = classify.class_mean(ds, true_labels)
    for l in np.unique(true_labels):
        assert means.where(true_labels == l).std() == 0
