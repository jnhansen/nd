import numpy as np
import xarray as xr
import re
from .. import utils


def disassemble_complex(ds, inplace=False):
    """Disassemble complex valued data into real and imag parts.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with complex variables.
    inplace : bool, optional
        Whether to modify the dataset inplace (default: False).

    Returns
    -------
    xarray.Dataset or None
        If inplace, returns None. Otherwise, returns a dataset where all
        complex variables have been split into their real and imaginary parts.
    """

    if isinstance(ds, xr.DataArray):
        name = ds.name
        if name is None:
            name = 'data'
        ds = ds.to_dataset(name=name)

    if inplace:
        new_ds = ds
    else:
        new_ds = ds.copy()
    #
    # Find all complex variables and disassemble into their real and
    # imaginary parts.
    #
    for vn, var in ds.data_vars.items():
        if not np.iscomplexobj(var):
            continue

        new_ds[vn + '__re'] = var.real
        new_ds[vn + '__im'] = var.imag
        del new_ds[vn]

    # reapply chunks
    if len(ds.chunks) > 0:
        new_ds = new_ds.chunk(ds.chunks)
    if not inplace:
        return new_ds


def assemble_complex(ds, inplace=False):
    """Reassemble complex valued data.

    NOTE: Changes the dataset (view) in place!

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with complex variables split into real and imaginary
        parts.
    inplace : bool, optional
        Whether to modify the dataset inplace (default: False).

    Returns
    -------
    xarray.Dataset or None
        If inplace, returns None. Otherwise, returns a dataset where the
        real and imaginary parts have been combined into the respective
        complex variables.
    """

    if inplace:
        new_ds = ds
    else:
        new_ds = ds.copy()
    # find all variables that are meant to be complex
    endings = {'re': ['_real', '__re'],
               'im': ['_imag', '__im']}
    rex = {}
    matches = {}
    for part, end in endings.items():
        rex[part] = re.compile('(?P<stem>.*)(?:{})'.format('|'.join(end)))
        matches[part] = [rex[part].match(vn) for vn in ds.data_vars]
        matches[part] = [_ for _ in matches[part] if _ is not None]

    new_var_names = set([m.group('stem') for m in
                         matches['re'] + matches['im']])

    for vn in new_var_names:
        vn_re = utils.select(matches['re'], lambda x: x.group(1) == vn,
                             first=True)
        vn_im = utils.select(matches['im'], lambda x: x.group(1) == vn,
                             first=True)
        if vn_re is not None and vn_im is not None:
            new_ds[vn] = new_ds[vn_re.group(0)] + new_ds[vn_im.group(0)] * 1j
            del new_ds[vn_re.group(0)]
            del new_ds[vn_im.group(0)]

    # reapply chunks
    if not inplace:
        # new_ds = new_ds.chunk(ds.chunks)
        return new_ds


def add_time(ds, inplace=False):
    """
    Add a `time` dimension to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.

    Returns
    -------
    xarray.Dataset
        A dataset that is guaranteed to contain the dimension `time`.
    """
    if not inplace:
        result = ds.copy()
    else:
        result = ds
    if 'time' not in result.coords:
        times = [utils.str2date(ds.attrs['start_date'])]
        result.coords['time'] = times
    if not inplace:
        return result
