"""
Quickly visualize datasets.

TODO: Update to work with xarray Dataset rather than GDAL.

"""
import os
import imageio
import cv2
import xarray as xr
import numpy as np


__all__ = ['colorize',
           'to_rgb',
           'write_video']

CMAPS = {
    'jet': cv2.COLORMAP_JET,
    'hsv': cv2.COLORMAP_HSV,
    'hot': cv2.COLORMAP_HOT,
    'cool': cv2.COLORMAP_COOL
}


def _cmap_from_str(cmap):
    if cmap in CMAPS:
        return CMAPS[cmap]
    else:
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
        else:
            # Both are None: Use original shape
            height = orig_shape[0]
            width = orig_shape[1]
    elif width is None:
        # Determine width from given height
        width = height * orig_shape[1] / orig_shape[0]

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
    data_color = cv2.applyColorMap(data_gray, _cmap_from_str(cmap))
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
                colored = cv2.applyColorMap(colored, _cmap_from_str(cmap))
        else:
            # im is in RGB
            colored = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # if output is not None:
        #     colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    if mask is not None:
        colored[~mask] = 0

    shape = calculate_shape(shape, colored.shape[:2])
    colored = cv2.resize(colored, shape)

    if output is None:
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        cv2.imwrite(output, colored)


def write_video(ds, path, timestamp=True, width=None, height=None, fps=1,
                codec=None, rgb=lambda d: [d.C11, d.C22, d.C11/d.C22],
                **kwargs):
    """
    Create a video from an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset must have dimensions 'y', 'x', and 'time'.
    path : str
        The output file path of the video.
    timestamp : bool, optional
        Whether to print the timestamp in the upper left corner
        (default: True).
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
        For a DataArray, the video will be grayscale.
    """
    # Font properties for timestamp
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 40)
    fontScale = 1
    fontColor = (0, 0, 0)
    lineType = 2

    # For a DataArray, the video is grayscale.
    if isinstance(ds, xr.DataArray):
        def rgb(d):
            return d

    # Use coords rather than dims so it also works for DataArray
    height, width = calculate_shape(
        (height, width), (ds.coords['y'].size, ds.coords['x'].size)
    )

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
            frame = to_rgb(rgb(d))
            frame = cv2.resize(frame, (width, height))
            if timestamp:
                cv2.putText(frame, str(t)[:10],
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
            writer.append_data(frame)
