import xarray as xr
import inspect
from . import utils
from . import warp
from . import change
from . import filters
from . import visualize
from . import io


def patch_doc(source):
    """Return a decorator that patched the docstring and signature of a
    class method with the corresponding data from `source`.
    """
    def _patch(func):
        # Override signature.
        # The first parameter is always the xarray object itself
        sig = inspect.signature(source)
        sig_extra = inspect.signature(func)
        extra_params = tuple(p for name, p in sig_extra.parameters.items()
                             if name not in ['self', 'args', 'kwargs'])
        self_param = inspect.signature(func).parameters['self']
        parameters = (self_param,) + tuple(sig.parameters.values())[1:] + \
            extra_params
        parameters = sorted(
            parameters,
            key=lambda p: (p.kind, p.default is not inspect._empty)
        )
        new_sig = sig.replace(parameters=parameters)
        func.__signature__ = new_sig

        # Override docstring
        doc = utils.parse_docstring(source.__doc__)
        doc_extra = utils.parse_docstring(func.__doc__)
        if 'Parameters' in doc:
            doc['Parameters'] = doc['Parameters'][1:]
        if 'Parameters' in doc_extra:
            if 'Parameters' not in doc:
                doc['Parameters'] = []
            doc['Parameters'] += doc_extra['Parameters']
        func.__doc__ = utils.assemble_docstring(doc, sig=new_sig)

        return func

    return _patch


@xr.register_dataset_accessor('nd')
@xr.register_dataarray_accessor('nd')
class NDAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # Properties
    @property
    def shape(self):
        return utils.get_shape(self._obj)

    @property
    def dims(self):
        return utils.get_dims(self._obj)

    @property
    def crs(self):
        return warp.get_crs(self._obj)

    @property
    def bounds(self):
        return warp.get_bounds(self._obj)

    @property
    def resolution(self):
        return warp.get_resolution(self._obj)

    @property
    def transform(self):
        return warp.get_transform(self._obj)

    # IO methods
    @patch_doc(io.assemble_complex)
    def as_complex(self):
        return io.assemble_complex(self._obj)

    @patch_doc(io.disassemble_complex)
    def as_real(self):
        return io.disassemble_complex(self._obj)

    @patch_doc(io.to_netcdf)
    def to_netcdf(self, *args, **kwargs):
        return io.to_netcdf(self._obj, *args, **kwargs)

    # General
    @patch_doc(utils.apply)
    def apply(self, *args, **kwargs):
        return utils.apply(self._obj, *args, **kwargs)

    # Visualization
    @patch_doc(visualize.to_rgb)
    def to_rgb(self, rgb=None, *args, **kwargs):
        """
        Parameters
        ----------
        rgb : callable
            A function returning an RGB tuple from the dataset,
            e.g., `lambda d: [d.B4, d.B3, d.B2]` or
            `lambda d: [d.isel(band=0), d.isel(band=1), d.isel(band=2)]`
        """
        if isinstance(self._obj, xr.DataArray) and rgb is None:
            data = self._obj
        else:
            if rgb is None:
                def rgb(d):
                    return[d.C11, d.C22, d.C11/d.C22]
            data = rgb(self._obj)
        return visualize.to_rgb(data, *args, **kwargs)

    @patch_doc(visualize.write_video)
    def to_video(self, *args, **kwargs):
        return visualize.write_video(self._obj, *args, **kwargs)

    @patch_doc(visualize.plot_map)
    def plot_map(self, *args, **kwargs):
        return visualize.plot_map(self._obj, *args, **kwargs)

    # Projection methods
    @patch_doc(warp.reproject)
    def reproject(self, *args, **kwargs):
        return warp.reproject(self._obj, *args, **kwargs)

    @patch_doc(warp.resample)
    def resample(self, *args, **kwargs):
        return warp.resample(self._obj, *args, **kwargs)

    # Change detection methods
    @patch_doc(change.omnibus)
    def change_omnibus(self, *args, **kwargs):
        return change.omnibus(self._obj, *args, **kwargs)


@xr.register_dataset_accessor('filter')
@xr.register_dataarray_accessor('filter')
class FilterAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # Filter methods
    @patch_doc(filters.nlmeans)
    def nlmeans(self, *args, **kwargs):
        return filters.nlmeans(self._obj, *args, **kwargs)

    @patch_doc(filters.boxcar)
    def boxcar(self, *args, **kwargs):
        return filters.boxcar(self._obj, *args, **kwargs)

    @patch_doc(filters.convolution)
    def convolve(self, *args, **kwargs):
        return filters.convolution(self._obj, *args, **kwargs)

    @patch_doc(filters.gaussian)
    def gaussian(self, *args, **kwargs):
        return filters.gaussian(self._obj, *args, **kwargs)
