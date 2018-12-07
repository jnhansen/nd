import xarray as xr


def open_rasterio(path, *args, **kwargs):
    return xr.open_rasterio(path, *args, **kwargs)
