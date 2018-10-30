import numpy as np
import re
from .. import utils


def disassemble_complex(ds):
    """Disassemble complex valued data into real and imag parts.

    Parameters
    ----------
    ds : xarray.Dataset
    """
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
    new_ds = new_ds.chunk(ds.chunks)
    return new_ds


def assemble_complex(ds, inplace=False):
    """Reassemble complex valued data.

    NOTE: Changes the dataset (view) in place!

    Parameters
    ----------
    ds : xarray.Dataset
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


def add_time(ds):
    result = ds.copy()
    if 'time' not in result.coords:
        times = [utils.str2date(ds.attrs['start_date'])]
        result.coords['time'] = times
    return result
