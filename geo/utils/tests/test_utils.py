import numpy as np
from numpy.testing import assert_equal
from geo import utils


def test_str2date():
    pass


def test_dict_product():
    d = {'a': [1, 2, 3], 'b': [5, 6]}
    result = [{'a': 1, 'b': 5},
              {'a': 1, 'b': 6},
              {'a': 2, 'b': 5},
              {'a': 2, 'b': 6},
              {'a': 3, 'b': 5},
              {'a': 3, 'b': 6}]
    assert_equal(list(utils.dict_product(d)), result)


def test_chunks():
    ll = np.arange(100)
    n = 15
    for i, c in enumerate(utils.chunks(ll, n)):
        assert_equal(c, ll[i*n:(i+1)*n])
