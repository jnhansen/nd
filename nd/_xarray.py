import xarray as xr
from . import warp


@xr.register_dataset_accessor('nd')
@xr.register_dataarray_accessor('nd')
class NDAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def reproject(self, *args, **kwargs):
        return warp.reproject(self._obj, *args, **kwargs)
