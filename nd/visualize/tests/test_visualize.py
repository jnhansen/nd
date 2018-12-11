from nd.visualize import to_rgb, write_video
from numpy.testing import assert_equal
from nd.testing import generate_test_dataset
import numpy as np
import pytest
import imageio


def _get_video_frame_count(path):
    video = imageio.mimread(path)
    return len(video)


def test_to_rgb_gray():
    N = 4
    values = np.arange(N**2).reshape((N, N))
    rgb = np.repeat(values[:, :, np.newaxis], 3, axis=2) * 255 / (N**2 - 1)
    assert_equal(
        to_rgb(values, pmin=0, pmax=100), rgb
    )


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


@pytest.mark.parametrize('path', [
    'video.gif',
    'video.avi',
    'video.mp4'
])
def test_write_video(tmpdir, path):
    ntime = 10
    ds = generate_test_dataset(ntime=ntime)
    write_video(ds, path)
    assert _get_video_frame_count(path) == ntime
