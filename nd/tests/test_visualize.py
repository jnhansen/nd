from nd.visualize import to_rgb, write_video, colorize, calculate_shape
from numpy.testing import assert_equal
from nd.testing import generate_test_dataset
import numpy as np
import pytest
import imageio
import os


@pytest.mark.parametrize('shape,orig_shape,expected', [
    (None, (40, 60), (40, 60)),
    ((None, None), (40, 60), (40, 60)),
    ((None, 120), (40, 60), (80, 120)),
    ((60, None), (40, 60), (60, 90)),
    ((110, 110), (40, 60), (110, 110)),
])
def test_calculate_shape(shape, orig_shape, expected):
    assert_equal(
        calculate_shape(shape, orig_shape),
        expected
    )


@pytest.mark.parametrize('output', [None, 'test.jpg'])
@pytest.mark.parametrize('pmin,pmax,vmin,vmax', [
    (0, 100, None, None),
    (2, 98, None, None),
    (None, None, 5, 90),
])
def test_to_rgb_gray(tmpdir, output, pmin, pmax, vmin, vmax):
    if output is not None:
        output = str(tmpdir / output)
    N = 10
    values = np.arange(N**2).reshape((N, N))

    if pmin is not None:
        rgb = np.clip(values, np.percentile(values, pmin),
                      np.percentile(values, pmax))
    elif vmin is not None:
        rgb = np.clip(values, vmin, vmax)

    # Generate grayscale image with values from 0..255
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
    rgb = np.repeat(rgb[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    result = to_rgb(values, output=output, pmin=pmin, pmax=pmax,
                    vmin=vmin, vmax=vmax)

    if output is None:
        assert_equal(result, rgb)
    else:
        assert os.path.isfile(output)


def test_to_rgb_color():
    N = 2
    values = [
        np.arange(N**2).reshape((N, N)),
        np.ones((N, N)),
        np.ones((N, N))
    ]
    rgb = np.stack(
        [v if v.max() == v.min() else (v - v.min()) / (v.max() - v.min()) * 255
         for v in values], axis=-1
    )
    assert_equal(
        to_rgb(values, pmin=0, pmax=100), rgb
    )


def test_to_rgb_invalid_datatype():
    with pytest.raises(ValueError):
        to_rgb('string')


@pytest.mark.parametrize('fname', [
    'video.gif',
    'video.avi',
    'video.mp4'
])
def test_write_video(tmpdir, fname):
    path = str(tmpdir.join(fname))
    ntime = 10
    ds = generate_test_dataset(dims={'y': 20, 'x': 20, 'time': ntime})
    write_video(ds, path)
    assert os.path.isfile(path)
    video = imageio.mimread(path)
    assert len(video) == ntime


@pytest.mark.parametrize('N', [10, None])
@pytest.mark.parametrize('nan_vals', [
    [], [3, 4, 5]
])
def test_colorize(N, nan_vals):
    nlabels = N if N is not None else 10
    labels = np.random.randint(0, nlabels, 400).reshape((20, 20))
    colored = colorize(labels, N=N, nan_vals=nan_vals)
    assert colored.shape == (20, 20, 3)
    for i in range(nlabels):
        # check that pixels of the same label have the same color
        mask = (labels == i)
        ncolors = len(np.unique(colored[mask], axis=0))
        assert ncolors == 1

    # Check that nan values are black
    for nv in nan_vals:
        assert (colored[labels == nv] == 0).all()
