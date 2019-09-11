import pytest
from nd.algorithm import (Algorithm, extract_arguments, wrap_algorithm)
from nd.testing import (generate_test_dataset, generate_test_dataarray)
from xarray.testing import assert_equal as xr_assert_equal
from numpy.testing import assert_raises_regex, assert_equal
from collections import OrderedDict
import inspect


# Create algorithm
class DummyAlgorithm(Algorithm):
    """test docstring"""
    def __init__(self, value, *args, **kwargs):
        self.value = value

    def apply(self, ds):
        return ds + self.value


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
def test_wrap_algorithm(generator):
    ds = generator()
    args = (0.1,)
    kwargs = {}
    algo = DummyAlgorithm(*args, **kwargs)
    wrapper = wrap_algorithm(DummyAlgorithm, 'wrapper_name')

    # Make sure the result is the same
    xr_assert_equal(
        algo.apply(ds),
        wrapper(ds, *args, **kwargs)
    )

    # Check name, docstring, signature
    assert(wrapper.__name__ == 'wrapper_name')
    assert_equal(
        wrapper.__doc__,
        'Wrapper for :class:`nd.tests.test_algorithm.DummyAlgorithm`.\n\n'
        + DummyAlgorithm.__doc__)
    assert_equal(
        list(OrderedDict(inspect.signature(wrapper).parameters).keys()),
        ['ds', 'value', 'args', 'kwargs']
    )


def test_invalid_algorithm_no_apply():
    class MissingApplyAlgorithm(Algorithm):
        def __init__(self, *args, **kwargs):
            pass

    # Check that the invalid algorithm cannot be instantiated
    with assert_raises_regex(
        TypeError,
        "Can't instantiate abstract class MissingApplyAlgorithm "
        "with abstract methods apply"
    ):
        MissingApplyAlgorithm()


@pytest.mark.parametrize('args,kwargs', [
    ((1, 2, 3), dict(c=4, d=5)),
    ((1,), dict(b=2, d=3)),
    ((1, 2, 3, 4, 5), dict()),
    ((), dict(b=2, a=1)),
])
def test_extract_arguments(args, kwargs):
    def fn(a, b, *args, c=None, **kwargs):
        return OrderedDict(
            a=a, b=b, args=args, c=c, kwargs=kwargs
        )

    bound = extract_arguments(fn, args, kwargs)
    actual = fn(*args, **kwargs)
    assert_equal(bound, actual)
