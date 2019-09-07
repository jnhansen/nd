import pytest
import numpy as np
from nd.classify import Classifier
from nd.testing import generate_test_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from numpy.testing import assert_equal


@pytest.mark.parametrize('clf', [
    GaussianNB(),
    KNeighborsClassifier(3)
])
def test_classifier(clf):
    ny, nx = 50, 50
    ds = generate_test_dataset(
        dims={'y': ny, 'x': nx},
        mean=[1, 0, 0, 1], sigma=0.1)
    ds2 = generate_test_dataset(
        dims={'y': ny, 'x': nx},
        mean=[10, 0, 0, 10], sigma=0.1)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[15:35, 15:35] = True
    ds = ds.where(mask, ds2)
    labels_true = np.ones((ny, nx))
    labels_true[mask] = 2

    # Select 10% for training
    labels_train = labels_true.copy()
    mask_train = (np.random.rand(ny, nx) < 0.9)
    labels_train[mask_train] = np.nan
    c = Classifier(clf)
    c.fit(ds, labels_train)
    labels_predicted = c.predict(ds).values

    # Expect 100% accuracy for this trivial classification task.
    assert_equal(labels_predicted, labels_true)
