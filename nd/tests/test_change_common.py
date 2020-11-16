import pytest
import xarray as xr
from numpy.testing import assert_equal
from xarray.testing import assert_equal as xr_assert_equal
from xarray.testing import assert_allclose as xr_assert_allclose
from xarray.testing import assert_identical as xr_assert_identical
import nd.change
from nd.testing import all_algorithms
from nd.change import ChangeDetection
from nd.testing import generate_test_dataset


ds = generate_test_dataset(dims={'y': 20, 'x': 30, 'time': 10})

change_detectors = [alg for alg in all_algorithms(nd.change)
                    if issubclass(alg[1], ChangeDetection) and
                    alg[0] != 'ChangeDetection']
cd_classes = [_[1] for _ in change_detectors]


@pytest.mark.parametrize('cd', cd_classes)
def test_change_input_output(cd):
    if cd._skip:
        pytest.skip('The dependency requirements for this class are '
                    'not fulfilled: {}'.format(cd._requires))

    instance = cd()
    result = instance.apply(ds)

    # Check that the output is an xarray Dataset
    assert isinstance(result, xr.DataArray)
    assert result.name == 'change'


# @pytest.mark.parametrize('cd', cd_classes)
# def test_change(cd):
#     instance = cd()
#     ds1 = generate_test_dataset(
#         mean=[1, 0, 0, 1], sigma=0.1, ny=5, nx=5
#         ).isel(time=slice(None, 5))
#     ds2 = generate_test_dataset(
#         mean=[10, 0, 0, 10], sigma=0.1, ny=5, nx=5
#         ).isel(time=slice(5, None))
#     ds = xr.concat([ds1, ds2], dim='time')
#     changes = instance.apply(ds)
#     print(changes.sum(dim='time'))
#     assert changes.isel(time=5).all()
#     assert (changes.sum(dim='time') == 1).all()
