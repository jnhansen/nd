import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from .transform import latlon_extent


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
    except:
        data = src

    ## RESAMPLE
    data_ = data[::N, ::N]

    plt.figure(figsize=(20, 20))
    plt.imshow(data_, vmin=0, vmax=255)
    plt.savefig(name)


def plot_basemap(src, name, *args, **kwargs):
    """
    A simple function to plot a (warped) gdal Dataset on a map.

    Parameters
    ----------
    src : osgeo.gdal.Dataset
        The (warped) dataset to plot.
    name : str
        The filename to save the plot, including extension.
    *args : list
    **kwargs : dict
        Will be passed on to pcolormesh(). E.g. vmin and vmax.

    """
    data = src.ReadAsArray()
    ndv = src.GetRasterBand(1).GetNoDataValue()
    data[data == ndv] = np.nan
    ## RESCALE to 2000px max dimension
    if max(data.shape) > 2000:
        new_shape = tuple(int(_) for _ in
                          np.array(data.shape) / max(data.shape) * 2000)
        data_ = skimage.transform.resize(data, new_shape, preserve_range=True)
    else:
        data_ = data
    print(data_.shape)

    extent = latlon_extent(src)
    fig = plt.figure(figsize=(10, 6), dpi=300)
    #ax = fig.add_axes([0.05,0.05,0.9,0.85])

    ## Plot data with a 20% margin around
    margin = 0.2
    llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = extent
    lon_extent = urcrnrlon - llcrnrlon
    lat_extent = urcrnrlat - llcrnrlat
    llcrnrlon -= margin * lon_extent
    urcrnrlon += margin * lon_extent
    llcrnrlat -= margin * lat_extent
    urcrnrlat += margin * lat_extent

    lat_0 = (urcrnrlat + llcrnrlat) / 2.0
    lon_0 = (urcrnrlon + llcrnrlon) / 2.0

    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                urcrnrlat=urcrnrlat, resolution='i', projection='tmerc',
                lat_0=lat_0, lon_0=lon_0)

    m.drawcoastlines()
    # m.fillcontinents(color='coral',lake_color='aqua')
    # m.drawmapboundary(fill_color='aqua')
    # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0, 360, 1), labels=[False, False, True, True])
    m.drawparallels(np.arange(-90, 90, 1), labels=[True, True, False, False])

    ## ADD DATA TO MAP
    xs = np.linspace(extent[0], extent[2], data_.shape[2])
    ys = np.linspace(extent[1], extent[3], data_.shape[1])
    lon_grid, lat_grid = np.meshgrid(xs, ys, copy=False)
    x, y = m(lon_grid, lat_grid)

    # cmap = plt.cm.gist_rainbow
    # cmap = plt.cm.jet
    # cmap = plt.cm.binary
    # cmap.set_under('1.0')
    # cmap.set_bad('0.8')

    # im = m.imshow(data_, extent=extent, cmap=cmap, vmin=0, vmax=255)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = 0
    if 'vmax' not in kwargs:
        kwargs['vmax'] = 255
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.cm.jet

    # rgba = np.stack([data_[1,:,:],data_[0,:,:],data_[1,:,:],255*np.ones(data_.shape[1:])],axis=-1) / 255.0
    # raveled_pixel_shape = (rgba.shape[0]*rgba.shape[1], rgba.shape[2])
    # color_tuple = rgba.transpose((1,0,2)).reshape(raveled_pixel_shape)

    bands = [0, 1, 0]
    vmax = [120, 250, 120]
    if vmax is None:
        normalize = np.nanmax(np.nanmax(data_[bands, :, :], axis=1), axis=1)
    else:
        normalize = vmax
    rgb = data_[bands, :, :-1]
    # rgb[np.isnan(rgb)] = 0.0
    raveled_pixel_shape = (rgb.shape[1]*rgb.shape[2], rgb.shape[0])
    color_tuple = rgb.transpose((1, 2, 0)) \
                     .reshape(raveled_pixel_shape) / normalize
    color_tuple[color_tuple < 0.0] = 0.0
    color_tuple[color_tuple > 1.0] = 1.0
    alpha = ~np.isnan(color_tuple).all(axis=1) * 1.0
    color_tuple = np.insert(color_tuple, 3, alpha, 1)


    # # colors = np.array([data_[0,:,:].flatten(),data_[1,:,:].flatten(),data_[0,:,:].flatten(), np.zeros(data_.shape[1:]).flatten()]) / 255
    # colors = np.stack([data_[1,:,:],data_[1,:,:],data_[1,:,:],255*np.ones(data_.shape[1:])],axis=-1).reshape(-1,4) / 255.0
    # # colors = np.stack([data_[1,:,:].flatten(),data_[1,:,:].flatten(),data_[1,:,:].flatten(),255*np.ones(data_.shape[1:]).flatten()],axis=-1) / 255.0
    # # .transpose([2,1,0])
    # colors[colors < 0.0] = 0.0
    # colors[colors > 1.0] = 1.0
    # # colors[np.isnan(colors)] = 0.0
    # colorTuple = tuple(colors.tolist())
    # # colorTuple = tuple(colors.transpose().tolist())
    # # im = m.pcolormesh(x,y, data_, **kwargs)

    # im = m.imshow(rgb.transpose((1,2,0)),extent=[extent[0],extent[2],extent[1],extent[3]])
    im = m.pcolormesh(lon_grid, lat_grid, data_[1, :, :], color=color_tuple,
                      latlon=True, **kwargs)
    # im = m.pcolormesh(x,y, data_[1,:,:], vmin=0, vmax=255, cmap=plt.cm.jet)
    # im = m.pcolormesh(x,y, _data, cmap=cmap, vmin=_data.min(), vmax=_data.max())
    # im = m.pcolormesh(x,y, data_small.T, cmap=cmap, vmin=0, vmax=255)
    cb = plt.colorbar(orientation='vertical', fraction=0.10, shrink=0.7)
    plt.savefig(name)
