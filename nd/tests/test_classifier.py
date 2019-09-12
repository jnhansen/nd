import pytest
import numpy as np
import xarray as xr
from nd import utils
from nd import classify
from nd.testing import generate_test_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from collections import OrderedDict


def create_mock_classes(dims):
    shape = (dims['y'], dims['x'])
    ds = generate_test_dataset(
        dims=dims,
        mean=[1, 0, 0, 1], sigma=0.1)
    ds2 = generate_test_dataset(
        dims=dims,
        mean=[10, 0, 0, 10], sigma=0.1)
    mask = np.zeros(shape, dtype=bool)
    mask = xr.DataArray(np.zeros(shape, dtype=bool),
                        dims=('y', 'x'),
                        coords=dict(y=ds.y, x=ds.x))
    # Make half of the data belong to each class
    mask[:, :dims['x']//2] = True
    ds = ds.where(mask, ds2)
    labels_true = (mask * 2).where(mask, 1)
    return ds, labels_true


@pytest.mark.parametrize('clf', [
    GaussianNB(),
    KNeighborsClassifier(3),
    RandomForestClassifier(n_estimators=20),
])
def test_classifier(clf):
    dims = OrderedDict([('y', 50), ('x', 50)])
    ds, labels_true = create_mock_classes(dims)

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
    blabels = classify._broadcast_labels(labels, ds, feature_dims)
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
