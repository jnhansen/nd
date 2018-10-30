"""
Quickly visualize datasets.

TODO: Update to work with xarray Dataset rather than GDAL.

"""
import cv2
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.basemap import Basemap
    import skimage.transform
except ImportError:
    pass


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
           categorical=False, mask=None, size=None, cmap=None):
    """
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
        highest percentile to plot (default: 2). Ignored if vmax is passed.
    """
    if isinstance(data, list):
        n_channels = len(data)
    elif isinstance(data, xr.DataArray) or isinstance(data, np.ndarray):
        n_channels = 1
        data = [data]
    else:
        raise ValueError("`data` must be a DataArray or list of DataArrays")

    values = [np.asarray(d) for d in data]
    shape = data[0].shape + (n_channels,)

    if vmin is not None:
        if isinstance(vmin, (int, float)):
            vmin = [vmin] * n_channels
    if vmax is not None:
        if isinstance(vmax, (int, float)):
            vmax = [vmax] * n_channels

    if categorical:
        colored = colorize(values[0], nan_vals=[0])

    else:
        im = np.empty(shape)

        for i in range(n_channels):
            channel = values[i]
            # Stretch
            if vmin is not None:
                minval = vmin[i]
            else:
                minval = np.percentile(channel, pmin)
            if vmax is not None:
                maxval = vmax[i]
            else:
                maxval = np.percentile(channel, pmax)
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

    if size is not None:
        if size[0] is None:
            size = (int(colored.shape[0] * size[1] / colored.shape[1]),
                    size[1])
        elif size[1] is None:
            size = (size[0],
                    int(colored.shape[1] * size[0] / colored.shape[0]))
        colored = cv2.resize(colored, (size[1], size[0]))

    if output is None:
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        cv2.imwrite(output, colored)


def plot_image(src, name, N=1):
    """
    A simple convenience function for plotting a GDAL dataset as an image.

    Parameters
    ----------
    src : osgeo.gdal.Dataset or np.ndarray
        The input data.
    name : str
        The filename including extension.
    N : int, opt
        The number
    """
    try:
        data = src.ReadAsArray()
    except AttributeError:
        data = src

    # RESAMPLE
    data_ = data[::N, ::N]

    plt.figure(figsize=(20, 20))
    plt.imshow(data_, vmin=0, vmax=255)
    plt.savefig(name)


# def plot_basemap(src, name, *args, **kwargs):
#     """
#     A simple function to plot a (warped) gdal Dataset on a map.

#     Parameters
#     ----------
#     src : osgeo.gdal.Dataset
#         The (warped) dataset to plot.
#     name : str
#         The filename to save the plot, including extension.
#     *args : list
#     **kwargs : dict
#         Will be passed on to pcolormesh(). E.g. vmin and vmax.

#     """
#     data = src.ReadAsArray()
#     ndv = src.GetRasterBand(1).GetNoDataValue()
#     data[data == ndv] = np.nan
#     # RESCALE to 2000px max dimension
#     if max(data.shape) > 2000:
#         new_shape = tuple(int(_) for _ in
#                           np.array(data.shape) / max(data.shape) * 2000)
#         data_ = skimage.transform.resize(data, new_shape, preserve_range=True)
#     else:
#         data_ = data
#     print(data_.shape)

#     extent = latlon_extent(src)
#     fig = plt.figure(figsize=(10, 6), dpi=300)
#     # ax = fig.add_axes([0.05,0.05,0.9,0.85])

#     # Plot data with a 20% margin around
#     margin = 0.2
#     llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = extent
#     lon_extent = urcrnrlon - llcrnrlon
#     lat_extent = urcrnrlat - llcrnrlat
#     llcrnrlon -= margin * lon_extent
#     urcrnrlon += margin * lon_extent
#     llcrnrlat -= margin * lat_extent
#     urcrnrlat += margin * lat_extent

#     lat_0 = (urcrnrlat + llcrnrlat) / 2.0
#     lon_0 = (urcrnrlon + llcrnrlon) / 2.0

#     m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
#                 urcrnrlat=urcrnrlat, resolution='i', projection='tmerc',
#                 lat_0=lat_0, lon_0=lon_0)

#     m.drawcoastlines()
#     # m.fillcontinents(color='coral',lake_color='aqua')
#     # m.drawmapboundary(fill_color='aqua')
#     # labels = [left,right,top,bottom]
#     m.drawmeridians(np.arange(0, 360, 1), labels=[False, False, True, True])
#     m.drawparallels(np.arange(-90, 90, 1), labels=[True, True, False, False])

#     # ADD DATA TO MAP
#     xs = np.linspace(extent[0], extent[2], data_.shape[2])
#     ys = np.linspace(extent[1], extent[3], data_.shape[1])
#     lon_grid, lat_grid = np.meshgrid(xs, ys, copy=False)
#     x, y = m(lon_grid, lat_grid)

#     # cmap = plt.cm.gist_rainbow
#     # cmap = plt.cm.jet
#     # cmap = plt.cm.binary
#     # cmap.set_under('1.0')
#     # cmap.set_bad('0.8')

#     # im = m.imshow(data_, extent=extent, cmap=cmap, vmin=0, vmax=255)
#     if 'vmin' not in kwargs:
#         kwargs['vmin'] = 0
#     if 'vmax' not in kwargs:
#         kwargs['vmax'] = 255
#     if 'cmap' not in kwargs:
#         kwargs['cmap'] = plt.cm.jet

#     # rgba = np.stack([data_[1,:,:],data_[0,:,:],data_[1,:,:],255*np.ones(
#     # data_.shape[1:])],axis=-1) / 255.0
#     # raveled_pixel_shape = (rgba.shape[0]*rgba.shape[1], rgba.shape[2])
#     # color_tuple = rgba.transpose((1,0,2)).reshape(raveled_pixel_shape)

#     bands = [0, 1, 0]
#     vmax = [120, 250, 120]
#     if vmax is None:
#         normalize = np.nanmax(np.nanmax(data_[bands, :, :], axis=1), axis=1)
#     else:
#         normalize = vmax
#     rgb = data_[bands, :, :-1]
#     # rgb[np.isnan(rgb)] = 0.0
#     raveled_pixel_shape = (rgb.shape[1]*rgb.shape[2], rgb.shape[0])
#     color_tuple = rgb.transpose((1, 2, 0)) \
#                      .reshape(raveled_pixel_shape) / normalize
#     color_tuple[color_tuple < 0.0] = 0.0
#     color_tuple[color_tuple > 1.0] = 1.0
#     alpha = ~np.isnan(color_tuple).all(axis=1) * 1.0
#     color_tuple = np.insert(color_tuple, 3, alpha, 1)

#     # # colors = np.array([data_[0,:,:].flatten(),data_[1,:,:].flatten(),
#     # data_[0,:,:].flatten(), np.zeros(data_.shape[1:]).flatten()]) / 255
#     # colors = np.stack([data_[1,:,:],data_[1,:,:],data_[1,:,:],255*np.ones(
#     # data_.shape[1:])],axis=-1).reshape(-1,4) / 255.0
#     # # colors = np.stack([data_[1,:,:].flatten(),data_[1,:,:].flatten(),
#     # data_[1,:,:].flatten(),255*np.ones(data_.shape[1:]).flatten()],axis=-1) / 255.0
#     # # .transpose([2,1,0])
#     # colors[colors < 0.0] = 0.0
#     # colors[colors > 1.0] = 1.0
#     # # colors[np.isnan(colors)] = 0.0
#     # colorTuple = tuple(colors.tolist())
#     # # colorTuple = tuple(colors.transpose().tolist())
#     # # im = m.pcolormesh(x,y, data_, **kwargs)

#     # im = m.imshow(rgb.transpose((1,2,0)),
#     # extent=[extent[0], extent[2], extent[1], extent[3]])
#     im = m.pcolormesh(lon_grid, lat_grid, data_[1, :, :], color=color_tuple,
#                       latlon=True, **kwargs)
#     # im = m.pcolormesh(x,y, data_[1,:,:], vmin=0, vmax=255, cmap=plt.cm.jet)
#     # im = m.pcolormesh(x,y, _data, cmap=cmap,
#     #                   vmin=_data.min(), vmax=_data.max())
#     # im = m.pcolormesh(x,y, data_small.T, cmap=cmap, vmin=0, vmax=255)
#     cb = plt.colorbar(orientation='vertical', fraction=0.10, shrink=0.7)
#     plt.savefig(name)
