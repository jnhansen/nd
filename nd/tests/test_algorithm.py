import pytest
from nd.algorithm import (Algorithm, wrap_algorithm,
                          parallelize)
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
        """Apply dummy algorithm."""
        return ds + self.value


class ParallelDummyAlgorithm(Algorithm):
    """test docstring"""

    def __init__(self, value, *args, **kwargs):
        self.value = value

    @parallelize
    def apply(self, ds):
        """Apply dummy algorithm."""
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


@pytest.mark.parametrize('generator', [
    generate_test_dataset,
    generate_test_dataarray
])
@pytest.mark.parametrize('njobs', [-1, 1, 2])
def test_parallelized_apply(generator, njobs):
    ds = generator()
    algo = ParallelDummyAlgorithm(3)
    ref = algo.apply(ds)
    result = algo.apply(ds, njobs=njobs)
    xr_assert_equal(ref, result)
