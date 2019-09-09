import xarray as xr
from . import warp
from . import filters
from . import visualize
from . import io


@xr.register_dataset_accessor('nd')
@xr.register_dataarray_accessor('nd')
class NDAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # IO methods
    def as_complex(self):
        return io.assemble_complex(self._obj)

    def as_real(self):
        return io.disassemble_complex(self._obj)

    def to_netcdf(self, *args, **kwargs):
        return io.to_netcdf(self._obj, *args, **kwargs)

    # Visualization
    def to_rgb(self, rgb=lambda d: [d.C11, d.C22, d.C11/d.C22],
               *args, **kwargs):
        data = rgb(self._obj)
        return visualize.to_rgb(data, *args, **kwargs)

    def to_video(self, *args, **kwargs):
        return visualize.write_video(self._obj, *args, **kwargs)

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
