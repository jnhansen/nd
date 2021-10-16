from nd import visualize
from nd import warp
from numpy.testing import assert_equal
from nd.testing import generate_test_dataset, requires
import numpy as np
import pytest
import imageio
import os
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
except (ModuleNotFoundError, ImportError):
    pass
import shapely.affinity
import cv2


@pytest.mark.parametrize('shape,orig_shape,expected', [
    (None, (40, 60), (40, 60)),
    ((None, None), (40, 60), (40, 60)),
    ((None, 120), (40, 60), (80, 120)),
    ((60, None), (40, 60), (60, 90)),
    ((110, 110), (40, 60), (110, 110)),
])
def test_calculate_shape(shape, orig_shape, expected):
    assert_equal(
        visualize.calculate_shape(shape, orig_shape),
        expected
    )


@pytest.mark.parametrize('s,cmap', [
    ('jet', cv2.COLORMAP_JET),
    ('turbo', cv2.COLORMAP_TURBO),
    ('hot', cv2.COLORMAP_HOT),
])
def test_parse_cmap(s, cmap):
    for name in [s, s.upper()]:
        assert visualize._parse_cmap(name) == cmap


@pytest.mark.parametrize('cmap', [
    cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT
])
def test_parse_cmap_cv2(cmap):
    assert visualize._parse_cmap(cmap) == cmap


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

    result = visualize.to_rgb(values, output=output, pmin=pmin, pmax=pmax,
                              vmin=vmin, vmax=vmax)

    if output is None:
        assert_equal(result, rgb)
    else:
        assert os.path.isfile(output)


@pytest.mark.parametrize('cmap', ['hot', 'turbo'])
@pytest.mark.parametrize('vmin, vmax', [(0, 1), (0.2, 0.8), (0.45, 0.55)])
def test_to_rgb_cmap(cmap, vmin, vmax):
    cm = visualize._parse_cmap(cmap)
    values = np.random.rand(100, 100)
    rgb = visualize.to_rgb(values, vmin=vmin, vmax=vmax, cmap=cmap)
    test = np.atleast_2d([0, 255]).astype(np.uint8)
    color_range = cv2.applyColorMap(test, cm)
    color_min = color_range[0, 0, ::-1]
    color_max = color_range[0, 1, ::-1]
    assert (rgb[values <= vmin] == color_min).all()
    assert (rgb[values >= vmax] == color_max).all()


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
        visualize.to_rgb(values, pmin=0, pmax=100), rgb
    )


def test_to_rgb_invalid_datatype():
    with pytest.raises(ValueError):
        visualize.to_rgb('string')


@pytest.mark.parametrize('fname', [
    'video.gif',
    'video.avi',
    'video.mp4'
])
def test_write_video(tmpdir, fname):
    path = str(tmpdir.join(fname))
    ntime = 10
    ds = generate_test_dataset(dims={'y': 20, 'x': 20, 'time': ntime})
    visualize.write_video(ds, path)
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
    colored = visualize.colorize(labels, N=N, nan_vals=nan_vals)
    assert colored.shape == (20, 20, 3)
    for i in range(nlabels):
        # Check that pixels of the same label have the same color
        mask = (labels == i)
        ncolors = len(np.unique(colored[mask], axis=0))
        assert ncolors == 1

    # Check that nan values are black
    for nv in nan_vals:
        assert (colored[labels == nv] == 0).all()


@requires('cartopy')
@pytest.mark.parametrize('extent', [
    [10, 20, -25, -15],
    [-5, 2, -2, 3]
])
def test_gridlines_with_labels(tmpdir, extent):
    tolerance = 0.05
    tol_lon = tolerance * (extent[1] - extent[0])
    tol_lat = tolerance * (extent[3] - extent[2])
    center_lon = (extent[0] + extent[1]) / 2
    center_lat = (extent[2] + extent[3]) / 2
    crs = ccrs.Orthographic(center_lon, center_lat)
    # Create figure
    plt.figure()
    ax = plt.axes(projection=crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    xmin, xmax, ymin, ymax = ax.get_extent()
    # Draw gridlines
    visualize.gridlines_with_labels(ax)
    # Need to trigger draw to get correct label positions
    plt.savefig(str(tmpdir / 'figure.pdf'))
    xticklabels = list(ax.get_xticklabels())
    yticklabels = list(ax.get_yticklabels())

    def _parse_label(text):
        i = text.find('°')
        value = float(text[:i])
        suffix = text[i+1:]
        return value, suffix

    def check_lon_suffix(lon, suffix):
        sign = np.sign(round(lon))
        if sign == 0:
            assert suffix == ''
        elif sign == 1:
            assert suffix == 'E'
        elif sign == -1:
            assert suffix == 'W'

    def check_lat_suffix(lat, suffix):
        sign = np.sign(round(lat))
        if sign == 0:
            assert suffix == ''
        elif sign == 1:
            assert suffix == 'N'
        elif sign == -1:
            assert suffix == 'S'

    assert len(xticklabels) > 0
    assert len(yticklabels) > 0

    # Check that ticks are placed correctly
    for tick in xticklabels:
        text = tick.get_text()
        pos = tick.get_position()
        lon, lat = ccrs.PlateCarree().transform_point(
            x=pos[0], y=pos[1], src_crs=ax.projection)
        value, suffix = _parse_label(text)
        check_lon_suffix(lon, suffix)
        assert np.abs(value - np.abs(lon)) < tol_lon

    for tick in yticklabels:
        text = tick.get_text()
        pos = tick.get_position()
        lon, lat = ccrs.PlateCarree().transform_point(
            x=pos[0], y=pos[1], src_crs=ax.projection)
        value, suffix = _parse_label(text)
        check_lat_suffix(lat, suffix)
        assert np.abs(value - np.abs(lat)) < tol_lat


@requires('cartopy')
def test_plot_map(tmpdir):
    extent = [-5, -2, 2, 3]
    buffer = 2
    ds = generate_test_dataset(extent=extent)
    ax = visualize.plot_map(ds, buffer=buffer, background=None, scalebar=True)

    # Check that extent contains full boundary with buffer
    buffered = shapely.affinity.scale(
        warp.get_geometry(ds), xfact=(buffer + 1.0), yfact=(buffer + 1.0))
    xmin, ymin, xmax, ymax = buffered.bounds

    # For some reason, the order of x and y is reversed
    # in the axis extent...
    ax_xmin, ax_xmax, ax_ymin, ax_ymax = ax.get_extent(crs=ccrs.PlateCarree())
    xtol = 0.1 * (xmax - xmin)
    ytol = 0.1 * (ymax - ymin)
    numtol = 1e-3
    assert ax_xmin <= xmin + numtol
    assert ax_xmin > xmin - xtol
    assert ax_xmax >= xmax - numtol
    assert ax_xmax < xmax + xtol
    assert ax_ymin <= ymin + numtol
    assert ax_ymin > ymin - ytol
    assert ax_ymax >= ymax - numtol
    assert ax_ymax < ymax + ytol


@requires('cartopy')
def test_plot_map_with_background(tmpdir):
    extent = [-10.0, 50.0, 0.0, 56.0]
    ds = generate_test_dataset(extent=extent)
    plt.figure(figsize=(8, 8))
    ax = visualize.plot_map(ds, buffer=1, background='_default', imscale=1)
    # Need to trigger rendering to get image
    plt.savefig(str(tmpdir / 'figure.pdf'))
    assert len(ax.get_images()) == 1
