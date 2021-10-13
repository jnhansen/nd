"""
Quickly visualize datasets.

"""
import os
import imageio
import cv2
import xarray as xr
import numpy as np
try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.geodesic as cgeo
    import cartopy.io.img_tiles as cimgt
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except ImportError:
    cartopy = None
from matplotlib import ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pyproj
import shapely.geometry
import shapely.affinity
from . import warp
from .utils import requires


__all__ = ['colorize',
           'to_rgb',
           'write_video',
           'plot_map']

CMAPS = {
    'jet': cv2.COLORMAP_JET,
    'hsv': cv2.COLORMAP_HSV,
    'hot': cv2.COLORMAP_HOT,
    'cool': cv2.COLORMAP_COOL
}


def _parse_cmap(cmap):
    if cmap in CMAPS:
        return CMAPS[cmap]
    try:
        return getattr(cv2, 'COLORMAP_{}'.format(cmap.upper()))
    except AttributeError:
        pass
    return cmap


def calculate_shape(new_shape, orig_shape):
    """Calculate a new image shape given the desired width and/or height.

    Parameters
    ----------
    new_shape : tuple
        The desired height and/or width of the image. Each may be None.
    orig_shape : tuple
        The shape of the original image. Will be used to calculate the
        width and height in case one or both are None.

    Returns
    -------
    tuple
        The output shape

    """
    if new_shape is None:
        return orig_shape

    height, width = new_shape
    if height is None:
        if width is not None:
            # Determine height from given width
            height = width * orig_shape[0] / orig_shape[1]
            height = height // 2 * 2
        else:
            # Both are None: Use original shape
            height = orig_shape[0]
            width = orig_shape[1]
    elif width is None:
        # Determine width from given height
        width = height * orig_shape[1] / orig_shape[0]
        width = width // 2 * 2

    return (int(height), int(width))


def colorize(labels, N=None, nan_vals=[], cmap='jet'):
    """
    Apply a color map to a map of integer labels.

    Parameters
    ----------
    labels : np.array, shape (M,N)
        The labeled image.
    N : int, optional
        The number of colors to use (default: 10)

    Returns
    -------
    np.array, shape (M,N,3)
        A colored image in BGR space, ready to be handled by OpenCV.
    """
    if N is None:
        N = min(10, len(np.unique(labels)))
    data = (labels % N) * (255/(N-1))
    data_gray = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    data_color = cv2.applyColorMap(data_gray, _parse_cmap(cmap))
    for nv in nan_vals:
        data_color[labels == nv] = 0
    # data_color[labels == MASK_VAL] = 255
    return data_color


def to_rgb(data, output=None, vmin=None, vmax=None, pmin=2, pmax=98,
           categorical=False, mask=None, shape=None, cmap=None):
    """Turn some data into a numpy array representing an RGB image.

    Parameters
    ----------
    data : list of DataArray
    output : str
        file path
    vmin : float or list of float
        minimum value, or list of values per channel (default: None).
    vmax : float or list of float
        maximum value, or list of values per channel (default: None).
    pmin : float
        lowest percentile to plot (default: 2). Ignored if vmin is passed.
    pmax : float
        highest percentile to plot (default: 98). Ignored if vmax is passed.
    categorical : bool, optional
        Whether the data is categorical. If True, return a randomly colorized
        image according to the data value (default: False).
    mask : np.ndarray, optional
        If specified, parts of the image outside of the mask will be black.
    shape : tuple, optional
        The output height and width (either or both may be None)
    cmap : opencv colormap, optional
        The colormap used to colorize grayscale data

    Returns
    -------
    np.ndarray or None
        Returns the generate RGB image if output is None, else returns None.
    """

    if isinstance(data, list):
        n_channels = len(data)
    elif isinstance(data, xr.DataArray) or isinstance(data, np.ndarray):
        n_channels = 1
        data = [data]
    else:
        raise ValueError("`data` must be a DataArray or list of DataArrays")

    for d in data:
        if len(d.shape) > 2:
            raise ValueError("The RGB channels must be two-dimensional. "
                             "Found dimensions {}".format(d.dims))

    values = [np.asarray(d) for d in data]
    shape_rgb = data[0].shape + (n_channels,)

    if vmin is not None:
        if isinstance(vmin, (int, float)):
            vmin = [vmin] * n_channels
    if vmax is not None:
        if isinstance(vmax, (int, float)):
            vmax = [vmax] * n_channels

    if categorical:
        colored = colorize(values[0], nan_vals=[0])

    else:
        im = np.empty(shape_rgb)

        for i in range(n_channels):
            channel = values[i]
            # Stretch
            if vmin is not None:
                minval = vmin[i]
            else:
                minval = np.nanpercentile(channel, pmin)
            if vmax is not None:
                maxval = vmax[i]
            else:
                maxval = np.nanpercentile(channel, pmax)
            if maxval > minval:
                channel = (channel - minval) / (maxval - minval) * 255

            im[:, :, i] = channel
        im = np.clip(im, 0, 255).astype(np.uint8)
        if n_channels == 1:
            colored = cv2.cvtColor(im[:, :, 0], cv2.COLOR_GRAY2BGR)
            if cmap is not None:
                # colored is now in BGR
                colored = cv2.applyColorMap(colored, _parse_cmap(cmap))
        else:
            # im is in RGB
            colored = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # if output is not None:
        #     colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    if mask is not None:
        colored[~mask] = 0

    shape = calculate_shape(shape, colored.shape[:2])
    # cv2.resize requires (width, height) rather than (height, width)
    colored = cv2.resize(colored, shape[::-1])

    if output is None:
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        cv2.imwrite(output, colored)


def write_video(ds, path, timestamp='upper left', fontcolor=(0, 0, 0),
                width=None, height=None, fps=1,
                codec=None, rgb=None,
                cmap=None, mask=None, contours=None,
                **kwargs):
    """
    Create a video from an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset must have dimensions 'y', 'x', and 'time'.
    path : str
        The output file path of the video.
    timestamp : str, optional
        Location to print the timestamp:
        ['upper left', 'lower left', 'upper right', 'lower right',
        'ul', 'll', 'ur', 'lr']
        Set to `None` to disable (default: 'upper left').
    fontcolor : tuple, optional
        RGB tuple for timestamp font color (default: (0, 0, 0), i.e., black).
    width : int, optional
        The width of the video (default: ds.dim['x'])
    height : int, optional
        The height of the video (default: ds.dim['y'])
    fps : int, optional
        Frames per second (default: 1).
    codec : str, optional
        fourcc codec (see http://www.fourcc.org/codecs.php)
    rgb : callable, optional
        A callable that takes a Dataset as input and returns a list of
        R, G, B channels. By default will compute the C11, C22, C11/C22
        representation.
        For a DataArray, use ``cmap``.
    cmap : str, optional
        For DataArrays only. Colormap used to colorize univariate data.
    mask : np.ndarray, optional
        If specified, parts of the image outside of the mask will be black.
    """

    if rgb is None:
        # For a DataArray, the video is grayscale.
        if isinstance(ds, xr.DataArray):
            def rgb(d):
                return d
        else:
            def rgb(d):
                return [d.C11, d.C22, d.C11/d.C22]

    # Use coords rather than dims so it also works for DataArray
    height, width = calculate_shape(
        (height, width), (ds.coords['y'].size, ds.coords['x'].size)
    )

    # Font properties for timestamp
    if timestamp in ['upper right', 'ur']:
        bottomLeftCornerOfText = (width-230, 40)
    elif timestamp in ['lower left', 'll']:
        bottomLeftCornerOfText = (20, height-20)
    elif timestamp in ['lower right', 'lr']:
        bottomLeftCornerOfText = (width-230, height-20)
    elif timestamp in ['upper left', 'ul']:
        bottomLeftCornerOfText = (20, 40)
    else:
        bottomLeftCornerOfText = (20, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = fontcolor
    lineType = 2

    _, ext = os.path.splitext(path)

    writer_kwargs = {
        'mode': 'I',
        'fps': fps,
    }
    writer_kwargs.update(kwargs)
    if ext != '.gif':
        writer_kwargs['macro_block_size'] = None
        writer_kwargs['ffmpeg_log_level'] = 'error'
        if codec is None:
            codec = 'libx264'
        writer_kwargs['codec'] = codec

    with imageio.get_writer(path, **writer_kwargs) as writer:
        for t in ds.time.values:
            d = ds.sel(time=t)
            frame = to_rgb(rgb(d), cmap=cmap, mask=mask)
            # Contours
            if contours is not None:
                frame = cv2.drawContours(
                    frame, contours, -1, (255, 255, 255), thickness=1)
            frame = cv2.resize(frame, (width, height))
            if timestamp not in [False, None]:
                cv2.putText(frame, str(t)[:10],
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
            writer.append_data(frame)


# -------
# Mapping
# -------

@requires('cartopy')
def gridlines_with_labels(ax, top=True, bottom=True, left=True,
                          right=True, fontsize=12, max_nlines=5, **kwargs):
    """
    Like :meth:`cartopy.mpl.geoaxes.GeoAxes.gridlines`, but will draw
    gridline labels for arbitrary projections.

    Parameters
    ----------
    ax : :class:`cartopy.mpl.geoaxes.GeoAxes`
        The :class:`GeoAxes` object to which to add the gridlines.
    top, bottom, left, right : bool, optional
        Whether or not to add gridline labels at the corresponding side
        of the plot (default: all True).
    fontsize : int, optional
        Tick label fontsize (default: 12).
    max_nlines : int, optional
        Maximum number of gridlines to plot per axis (default: 5).
    kwargs : dict, optional
        Extra keyword arguments to be passed to :meth:`ax.gridlines`.

    Returns
    -------
    :class:`cartopy.mpl.gridliner.Gridliner`
        The :class:`Gridliner` object resulting from ``ax.gridlines()``.

    """
    # Add gridlines
    gridliner = ax.gridlines(**kwargs)
    gridliner.xlocator = mticker.MaxNLocator(max_nlines)
    gridliner.ylocator = mticker.MaxNLocator(max_nlines)

    ax.tick_params(length=0)

    # Get projected extent
    xmin, xmax, ymin, ymax = ax.get_extent()

    # Determine tick positions
    sides = {}
    N = 500
    if bottom:
        sides['bottom'] = np.stack([np.linspace(xmin, xmax, N),
                                    np.ones(N) * ymin])
    if top:
        sides['top'] = np.stack([np.linspace(xmin, xmax, N),
                                np.ones(N) * ymax])
    if left:
        sides['left'] = np.stack([np.ones(N) * xmin,
                                  np.linspace(ymin, ymax, N)])
    if right:
        sides['right'] = np.stack([np.ones(N) * xmax,
                                   np.linspace(ymin, ymax, N)])

    # Get latitude and longitude coordinates of axes boundary at each side
    # in discrete steps
    gridline_coords = {}
    for side, values in sides.items():
        gridline_coords[side] = ccrs.PlateCarree().transform_points(
            ax.projection, values[0], values[1])

    # Get longitude and latitude limits
    points = np.concatenate(list(gridline_coords.values()))
    lon_lim = (points[:, 0].min(), points[:, 0].max())
    lat_lim = (points[:, 1].min(), points[:, 1].max())

    ticklocs = {
        'x': gridliner.xlocator.tick_values(lon_lim[0], lon_lim[1]),
        'y': gridliner.ylocator.tick_values(lat_lim[0], lat_lim[1])
    }

    # Compute the positions on the outer boundary where
    coords = {}
    for name, g in gridline_coords.items():
        if name in ('bottom', 'top'):
            compare, axis = 'x', 0
        else:
            compare, axis = 'y', 1
        coords[name] = np.array([
            sides[name][:, np.argmin(np.abs(
                gridline_coords[name][:, axis] - c))]
            for c in ticklocs[compare]
        ])

    # Create overlay axes for top and right tick labels
    ax_topright = ax.figure.add_axes(ax.get_position(), frameon=False)
    ax_topright.tick_params(
        left=False, labelleft=False,
        right=True, labelright=True,
        bottom=False, labelbottom=False,
        top=True, labeltop=True,
        length=0
    )
    ax_topright.set_xlim(ax.get_xlim())
    ax_topright.set_ylim(ax.get_ylim())

    for side, tick_coords in coords.items():
        if side in ('bottom', 'top'):
            axis, idx = 'x', 0
        else:
            axis, idx = 'y', 1

        _ax = ax if side in ('bottom', 'left') else ax_topright

        ticks = tick_coords[:, idx]

        valid = np.logical_and(
            ticklocs[axis] >= gridline_coords[side][0, idx],
            ticklocs[axis] <= gridline_coords[side][-1, idx])

        if side in ('bottom', 'top'):
            _ax.set_xticks(ticks[valid])
            _ax.set_xticklabels([LONGITUDE_FORMATTER.format_data(t)
                                 for t in ticklocs[axis][valid]],
                                fontdict={'fontsize': fontsize})
        else:
            _ax.set_yticks(ticks[valid])
            _ax.set_yticklabels([LATITUDE_FORMATTER.format_data(t)
                                 for t in ticklocs[axis][valid]],
                                fontdict={'fontsize': fontsize})

    return gridliner


def _get_orthographic_projection(ds):
    center_lon, center_lat = \
        warp.get_geometry(ds).centroid.coords[0]
    crs = ccrs.Orthographic(center_lon, center_lat)
    return crs


def _get_scalebar_length(ax):
    extent = ax.get_extent()
    length = (extent[1] - extent[0]) / 1e3
    s = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    scale = max(s[s < length/5])
    return scale


@requires('cartopy')
def plot_map(ds, buffer=None, background='_default', imscale=6,
             gridlines=True, coastlines=True, scalebar=True,
             gridlines_kwargs={}):
    """
    Show the boundary of the dataset on a visually appealing map.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The dataset whose bounds to plot on the map.
    buffer : float, optional
        Margin around the bounds polygon to plot, relative to the polygon
        dimension. By default, add around 20% on each side.
    background : :class:`cartopy.io.img_tiles` image tiles, optional
        The basemap to plot in the background (default: Stamen terrain).
        If None, do not plot a background map.
    imscale : int, optional
        The zoom level of the background image (default: 6).
    gridlines : bool, optional
        Whether to plot gridlines (default: True).
    coastlines : bool, optional
        Whether to plot coastlines (default: True).
    scalebar : bool, optional
        Whether to add a scale bar (default: True).
    gridlines_kwargs : dict, optional
        Additional keyword arguments for gridlines_with_labels().

    Returns
    -------
    :class:`cartopy.mpl.geoaxes.GeoAxes`
        The corresponding GeoAxes object.

    """
    if background == '_default':
        try:
            background = cimgt.Stamen('terrain-background')
        except AttributeError:
            # cartopy < 0.17.0
            background = cimgt.StamenTerrain()

    # Get polygon shape
    # -----------------
    geometry_data = shapely.geometry.box(*ds.nd.bounds)
    if buffer is None:
        buffer = 1.2
    else:
        buffer += 1.0
    buffered = shapely.affinity.scale(
        geometry_data, xfact=buffer, yfact=buffer)
    project = pyproj.Transformer.from_crs(
            ds.nd.crs, 'epsg:4326')
    b = shapely.ops.transform(project.transform, buffered).bounds
    extent = [b[0], b[2], b[1], b[3]]
    bb = Bbox.from_extents(extent)

    # Define Orthographic map projection
    # (centered at the polygon)
    # ----------------------------------
    map_crs = _get_orthographic_projection(ds)
    proj4_params = map_crs.proj4_params
    if 'a' in proj4_params:
        # Some version of cartopy add the parameter 'a'.
        # For some reason, the CRS cannot be parsed by rasterio with
        # this parameter present.
        del proj4_params['a']

    # Create figure
    # -------------
    ax = plt.axes(xlim=(b[0], b[2]), ylim=(b[1], b[3]), projection=map_crs,
                  aspect='equal', clip_box=bb)
    ax.set_global()
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.apply_aspect()

    # Add additional map features
    # ---------------------------

    if background is not None:
        ax.add_image(background, imscale)

    if coastlines:
        color = 'black' if background is None else 'white'
        ax.coastlines(resolution='10m', color=color)

    if scalebar:
        # Determine optimal length
        scale = _get_scalebar_length(ax)
        scale_bar(ax, (0.05, 0.05), scale, linewidth=5, ha='center')

    # Add polygon
    # -----------
    geometry_map = warp.get_geometry(ds, crs=proj4_params)
    ax.add_geometries([geometry_map], crs=map_crs,
                      facecolor=(1, 0, 0, 0.2), edgecolor=(0, 0, 0, 1))

    if gridlines:
        color = '0.5' if background is None else 'white'
        gridlines_with_labels(ax, color=color, **gridlines_kwargs)

    return ax


# -------------------------------------------------------------
# CARTOPY SCALE BAR
# Taken from StackOverflow https://stackoverflow.com/a/50674451
# -------------------------------------------------------------

def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Parameters
    ----------
    start : tuple
        Starting point for the line.
    direction : np.ndarray, shape (2, 1)
        A direction vector.
    distance : float
        Positive distance to go past.
    dist_func : callable
        A two-argument function which returns distance.

    Returns
    -------
    np.ndarray, shape (2, 1)
        Coordinates of a point.

    """

    if distance <= 0:
        raise ValueError(
            "Minimum distance is not positive: {}".format(distance))

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Parameters
    ----------
    start : tuple
        Starting point for the line.
    end : tuple
        Outer bound on point's location.
    distance : float
        Positive distance to travel.
    dist_func : callable
        Two-argument function which returns distance.
    tol : float
        Relative error in distance to allow.

    Returns
    -------
    np.ndarray, shape (2, 1)
        Coordinates of a point.

    """

    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError("End is closer to start ({}) than "
                         "given distance ({}).".format(
                             initial_distance, distance
                         ))

    if tol <= 0:
        raise ValueError("Tolerance is not positive: {}".format(tol))

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Parameters
    ----------
    ax : :class:`cartopy.mpl.geoaxes.GeoAxes`
        Cartopy axes.
    start : tuple
        Starting point for the line in axes coordinates.
    distance : float
        Positive physical distance to travel.
    angle : float, optional
        Anti-clockwise angle for the bar, in radians. Default: 0
    tol : float, optional
        Relative error in distance to allow. Default: 0.01

    Returns
    -------
    np.ndarray, shape (2, 1)
        Coordinates of a point

    """

    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        inv = geodesic.inverse(a_phys, b_phys)
        try:
            # Geodesic().inverse returns a NumPy MemoryView like [[distance,
            # start azimuth, end azimuth]].
            return inv.base[0, 0]
        except TypeError:
            # In newer versions, it is a plain numpy array
            return inv[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to Cartopy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Parameters
    ----------
    ax : :class:`cartopy.mpl.geoaxes.GeoAxes`
        Cartopy axes.
    location : tuple
        Position of left-side of bar in axes coordinates.
    length : float
        Geodesic length of the scale bar.
    metres_per_unit : int, optional
        Number of metres in the given unit. (Default: 1000)
    unit_name : str, optional
        Name of the given unit. (Default: 'km')
    tol : float, optional
        Allowed relative error in length of bar. (Default: 0.01)
    angle : float, optional
        Anti-clockwise rotation of the bar.
    color : str, optional
        Color of the bar and text. (Default: 'black')
    linewidth : float, optional
        Same argument as for plot.
    text_offset : float, optonal
        Perpendicular offset for text in axes coordinates.
        (Default: 0.005)
    ha : str, optional
        Horizontal alignment. (Default: 'center')
    va : str, optional
        Vertical alignment. (Default: 'bottom')
    **plot_kwargs : dict, optional
        Keyword arguments for plot, overridden by **kwargs.
    **text_kwargs : dict, optional
        Keyword arguments for text, overridden by **kwargs.
    **kwargs : dict, optional
        Keyword arguments for both plot and text.

    """

    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, "{} {}".format(length, unit_name),
            rotation_mode='anchor', transform=ax.transAxes, **text_kwargs)
