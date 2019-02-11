import numpy as np
import xarray as xr
from ..algorithm import Algorithm
from ..utils import get_vars_for_dims
from ..io import disassemble_complex
import skimage


class Coregistration(Algorithm):
    """Coregister a time series (stack) of images to a master image.

    At the moment only supports coregistration by translation.

    Parameters
    ----------
    reference : int, optional
        The time index to use as reference for coregistration (default: 0).
    upsampling : int, optional
        The upsampling factor for shift estimation (default: 10).
    """

    def __init__(self, reference=0, upsampling=10, **kwargs):
        self.reference = reference
        self.upsampling = upsampling
        self.kwargs = kwargs

    def apply(self, ds):
        """Apply the projection to a dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset.

        Returns
        -------
        xarray.Dataset
            The coregistered dataset.
        """
        return _coregister(ds, reference=self.reference,
                           upsampling=self.upsampling)


def _coregister(ds, reference, upsampling, order=3):
    ref_var = 'C11'
    ds_new = disassemble_complex(ds)
    # disassemble_complex(ds_new, inplace=True)
    ref = ds_new.isel(time=reference)[ref_var].values
    datavars = get_vars_for_dims(ds_new, ['time', 'x', 'y'])
    print(datavars)
    # Coregister each time step independently
    for t in range(ds_new.dims['time']):
        if t == reference:
            continue
        # tval = ds['time'][t]
        src = ds_new.isel(time=t)
        # Estimate shift
        shift = skimage.feature.register_translation(
            src[ref_var].values, ref, upsample_factor=upsampling)
        translation = (shift[0][1], shift[0][0])
        print(translation)
        # Create transform object
        transf = skimage.transform.AffineTransform(translation=translation)
        # Apply transform to each variable
        for v in datavars:
            # For assignment, need to use .loc rather than .isel
            ds_new[v].loc[dict(time=ds['time'][t])] = skimage.transform.warp(
                src[v].values, transf, order=order)
    return ds_new
