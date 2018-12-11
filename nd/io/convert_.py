import numpy as np
import xarray as xr
import re
from .. import utils


def disassemble_complex(ds):
    """Disassemble complex valued data into real and imag parts.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with complex variables.

    Returns
    -------
    xarray.Dataset
        A dataset where all complex variables have been split into their real
        and imaginary parts.
    """

    if isinstance(ds, xr.DataArray):
        name = ds.name
        if name is None:
            name = 'data'
        ds = ds.to_dataset(name=name)

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
        The input dataset with complex variables split into real and imaginary
        parts.

    Returns
    -------
    xarray.Dataset
        A dataset where the real and imaginary parts have been combined into
        the respective complex variables.
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


def dualpol_to_complex(ds, inplace=False):
    """Convert a dataset containing the bands 'i_VV.img', 'q_VV.img',
    'i_VH.img', 'q_VH.img' into a dataset containing complex valued bands
    'VV' and 'VH'.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset in dual polarized format. Must contain the bands
        'i_VV.img', 'q_VV.img', 'i_VH.img', 'q_VH.img'.
    inplace : bool, optional
        If False, create a copy of the dataset, otherwise alter the dataset
        inplace (default: False).

    Returns
    -------
    xarray.Dataset
        A dataset with complex-valued variables 'VV' and 'VH'.
    """

    if inplace:
        ds_c = ds
    else:
        ds_c = ds.copy()
    dims = ds_c['i_VV.img'].dims
    vv = np.stack([ds_c['i_VV.img'], ds_c['q_VV.img']], axis=-1)
    vh = np.stack([ds_c['i_VH.img'], ds_c['q_VH.img']], axis=-1)
    ds_c['VV'] = (dims, vv.view(dtype=np.complex64)[:, :, 0])
    ds_c['VH'] = (dims, vh.view(dtype=np.complex64)[:, :, 0])
    del ds_c['i_VV.img']
    del ds_c['q_VV.img']
    del ds_c['i_VH.img']
    del ds_c['q_VH.img']
    if not inplace:
        return ds_c


def generate_covariance_matrix(ds, compact=False):
    """Convert from the canonical complex representation to the covariance
    matrix representation.

    Parameters
    ----------
    ds : xarray.Dataset
        A dual polarization dataset, containing two complex variables
        ``VH`` and ``VV``.
    compact : bool, optional
        If True, return a compact real representation. (default: False)

    Returns
    -------
    numpy.array, shape (M, N, 2, 2)
        The covariance matrix representation of the data.
    """

    if isinstance(ds, xr.Dataset):
        ds_cov = ds.copy()
        vh = ds['VH']
        vv = ds['VV']
    else:
        vv = ds[:, :, 0]
        vh = ds[:, :, 1]

    shape = vh.shape + (2, 2)

    if isinstance(ds, xr.Dataset):
        ds_cov['C11'] = np.real(vv * np.conj(vv))
        ds_cov['C22'] = np.real(vh * np.conj(vh))
        ds_cov['C12'] = vv * np.conj(vh)
        # NOTE: C21 is np.conj(C12) and hence is redundant information
        # and need not be stored.
        del ds_cov['VH']
        del ds_cov['VV']
        return ds_cov

    else:
        #
        # NOTE: The following is the legacy code for numpy arrays.
        #
        if compact:
            cov = np.empty(shape, dtype=np.float32)
        else:
            cov = np.empty(shape, dtype=np.complex64)
        cov[:, :, 0, 0] = vv * np.conj(vv)
        cov[:, :, 1, 1] = vh * np.conj(vh)
        C_12 = vv * np.conj(vh)
        if compact:
            cov[:, :, 0, 1] = np.real(C_12)
            cov[:, :, 1, 0] = np.imag(C_12)
        else:
            # C_12
            cov[:, :, 0, 1] = C_12
            cov[:, :, 1, 0] = np.conj(C_12)
        return cov
