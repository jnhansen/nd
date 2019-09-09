import xarray as xr
from . import warp
from . import filters
from .io import assemble_complex, disassemble_complex, to_netcdf


@xr.register_dataset_accessor('nd')
@xr.register_dataarray_accessor('nd')
class NDAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # IO methods
    def as_complex(self):
        return assemble_complex(self._obj)

    def as_real(self):
        return disassemble_complex(self._obj)

    def to_netcdf(self, *args, **kwargs):
        return to_netcdf(self._obj, *args, **kwargs)

    # Projection methods
    def reproject(self, *args, **kwargs):
        return warp.reproject(self._obj, *args, **kwargs)


@xr.register_dataset_accessor('filter')
@xr.register_dataarray_accessor('filter')
class FilterAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # Filter methods
    def nlmeans(self, *args, **kwargs):
        return filters.NLMeansFilter(*args, **kwargs).apply(self._obj)

    def boxcar(self, *args, **kwargs):
        return filters.BoxcarFilter(*args, **kwargs).apply(self._obj)

    def convolve(self, *args, **kwargs):
        return filters.ConvolutionFilter(*args, **kwargs).apply(self._obj)

    def gaussian(self, *args, **kwargs):
        return filters.GaussianFilter(*args, **kwargs).apply(self._obj)
