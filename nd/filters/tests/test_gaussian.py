from nd.filters import gaussian
import scipy.ndimage.filters as snf
from nd.testing import generate_test_dataset
from numpy.testing import assert_equal


ds = generate_test_dataset()


def test_gaussian_scalar_sigma():
    sigma = 1
    ds_gauss = gaussian(ds, sigma, dims=('lat', 'lon', 'time'))
    values_gauss = snf.gaussian_filter(ds.C11.values, sigma=sigma)
    assert_equal(ds_gauss.C11.values, values_gauss)


def test_gaussian_ignore_dimensions():
    # Make sure that dimensions which are not listed in `dims`
    # are not used for the Gaussian filter.
    sigma = 1
    ds_gauss = gaussian(ds, sigma, dims=('lat', 'lon')).isel(time=0)
    values_gauss = snf.gaussian_filter(
        ds.C11.isel(time=0).values, sigma=sigma
    )
    assert_equal(ds_gauss.C11.values, values_gauss)
