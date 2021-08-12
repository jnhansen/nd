"""
This module may be used to mosaic and tile multiple satellite image products.

TODO: Contain buffer information in NetCDF metadata?

"""
from .io import open_netcdf, to_netcdf, add_time
from . import utils
import os
import glob
import itertools
from functools import partial
import numpy as np
import xarray as xr
from dask import delayed


def tile(ds, path, prefix='part', chunks=None, buffer=0):
    """Split dataset into tiles and write to disk. If `chunks` is not given,
    use chunks in dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to split into tiles.
    path : str
        The output directory in which to place the tiles.
    prefix : str, optional
        The tile names will start with ```{prefix}.```
    chunks : dict, optional
        A dictionary of the chunksize for every dimension along which to split.
        By default, use the dask array chunksize if the data is represented as
        dask arrays.
    buffer : int or dict, optional
        The number of overlapping pixels to store around each tile
        (default: 0). Can be given as an integer or per dimension as
        dictionary.
    """

    # Prepare output directory
    if os.path.isfile(path):
        raise ValueError("`path` cannot be a file!")
    elif not os.path.isdir(path):
        os.makedirs(path)

    #  Treat `ds` as a file path
    if isinstance(ds, str):
        ds = xr.open_dataset(ds, engine='h5netcdf')

    # Prepare chunk sizes
    if chunks is None:
        chunked = ds
    else:
        chunked = ds.chunk(chunks)

    # 1. Convert chunk sizes into slice objects.
    slices = {}
    for dim, chunk_lens in chunked.chunks.items():
        start = 0
        slices[dim] = []

        if isinstance(buffer, int):
            _buf = buffer
        elif isinstance(buffer, dict) and dim in buffer:
            _buf = buffer[dim]
        else:
            _buf = 0

        for l in chunk_lens:
            # Apply buffer
            _start = max(0, start - _buf)
            slices[dim].append(
                slice(_start, start + l + _buf)
            )
            start += l

    #
    # Assume that the original chunks (ds.chunks) corresponds
    # to the natural splitting in files. Hence, optimize for
    # handling a single file at a time.
    #
    # ordered_keys = sorted(ds.chunks.keys(), key=lambda k: -len(ds.chunks[k]))
    # ordered_slices = OrderedDict()
    # for k in ordered_keys:
    #     ordered_slices[k] = slices[k]

    def _write_tile(slice_dict):
        # Slice the dataset and write to disk.
        subset = ds.isel(slice_dict)
        suffix = '.'.join(
            ['{}_{}_{}'.format(dim, s.start, s.stop)
             for dim, s in slice_dict.items()]
        )
        tile_name = '{}.{}.nc'.format(prefix, suffix)
        # Skip existing files.
        tile_path = os.path.join(path, tile_name)
        if not os.path.isfile(tile_path):
            temp_tile_path = tile_path + '.part'
            to_netcdf(subset, temp_tile_path)
            os.rename(temp_tile_path, tile_path)

    # 2. Then apply itertools to the slices.
    for slice_dict in utils.dict_product(slices):
        _write_tile(slice_dict)

    return


def map_over_tiles(files, fn, args=(), kwargs={}, path=None, suffix='',
                   merge=True, overwrite=False, compute=True):
    """Apply function to each tile.

    Parameters
    ----------
    files : str or list of str
        A glob expression matching all tiles. May also be a list of file paths.
    fn : function
        The function to apply to each tile.
    args : tuple, optional
        Additional arguments to fn
    kwargs : dict, optional
        Additional keyword arguments to fn
    path : str, optional
        The output directory. If None, write to same directory as input files.
        (default: None)
    suffix : str, optional
        If input file is `part.0.nc`, will create `part.0{suffix}.nc`.
        (default: '')
    merge : bool, optional
        If True, return a merged view of the result (default: True).
    overwrite : bool, optional
        Force overwriting existing files (default: False).
    compute : bool, optional
        If True, compute result immediately. Otherwise, return a dask.delayed
        object (default: True).

    Returns
    -------
    xarray.Dataset
        The (merged) output dataset.
    """

    if isinstance(files, str):
        files = glob.glob(files)

    if path is not None:
        os.makedirs(path, exist_ok=True)

    def _wrapper(f):
        # 1. Open dataset
        data = open_netcdf(f)
        # 2. Apply function
        result = fn(data, *args, **kwargs)
        # 3. Write result to file
        root, name = os.path.split(f)
        stem, ext = os.path.splitext(name)
        out_name = '{}{}{}'.format(stem, suffix, ext)
        out_path = root if path is None else path
        out_file = os.path.join(out_path, out_name)
        if not overwrite and os.path.exists(out_file):
            out_file = '{}_new{}'.format(*os.path.splitext(out_file))
        to_netcdf(result, out_file)
        # Garbage collect open datasets
        data.close()
        result.close()
        # 4. Return result
        return out_file

    results = [delayed(_wrapper)(f) for f in files]

    if merge:
        result = delayed(auto_merge)(results)
    else:
        result = delayed(results)

    if compute:
        return result.compute()
    else:
        return result


def sort_key(ds, dims):
    """To be used as key when sorting datasets."""
    keys = []
    for d in dims:
        vals = ds[d].values
        if len(vals) < 2 or vals[-1] >= vals[0]:
            # ascending order
            keys.append(vals[0])
            keys.append(vals[-1])
        else:
            # descending order
            keys.append(-vals[0])
            keys.append(-vals[-1])
    return tuple(keys)


# def _detect_buffer(datasets):
#     dims = utils.get_dims(datasets[0])
#     buffer = {}
#     for dim in dims:
#         _sorted = sorted(datasets, key=lambda ds: sort_key(ds, [dim]))
#         buffer[dim] = min([
#             len(a[dim] == b[dim]) for a, b in zip(_sorted[:-1], _sorted[1:])
#         ]) // 2
#     return buffer


def sort_into_array(datasets, dims=None):
    """
    Create an array corresponding to the way the datasets are tiled.
    """
    dims = utils.get_dims(datasets[0])
    initials = {}
    for dim in dims:
        initials[dim] = np.unique([d[dim].values[0] for d in datasets])
    shape = tuple(len(initials[dim]) for dim in dims)
    grid = np.empty(shape, dtype=object)

    def _idx(ds):
        result = []
        for dim in dims:
            vals = ds[dim].values
            if len(vals) < 2 or vals[-1] >= vals[0]:
                # ascending order
                order = 1
            else:
                # descending order
                order = -1

            result.append(
                np.argmax(initials[dim][::order] == ds[dim].values[0])
            )

        return tuple(result)

    for d in datasets:
        grid[_idx(d)] = d

    return grid


def debuffer(datasets, flat=True):
    """
    Remove buffer from tiled datasets.

    Parameters
    ----------
    datasets : list of xr.Dataset
        The overlapping tiles.
    flat : bool, optional
        If ``True``, return a flat list. Otherwise, return a numpy array
        representing the correct order of the tiles (default: True).
    """

    def _remove_buffer(data, dim):
        # data is already sorted by `dim`
        overlap = [len(a[dim] == b[dim])
                   for a, b in zip(data[:-1], data[1:])]
        buf_start = [o // 2 for o in overlap]
        buf_stop = [-(o-b) if b > 0 else None
                    for b, o in zip(buf_start, overlap)]
        debuf = [d.isel({dim: slice(start, stop)}) for d, start, stop
                 in zip(data, [None] + buf_start, buf_stop + [None])]
        # Force conversion to ndarray *without* converting the
        # xarray Datasets:
        # For numpy >= 1.20.0, the following line no longer works:
        #   return np.asarray(debuf + [None])[:-1]
        # Need to use the following hacky workaround:
        N = len(debuf)
        arr = np.empty(N, dtype=object)
        for i in range(N):
            arr[i] = debuf[i]
        return arr

    dims = utils.get_dims(datasets[0])
    grid = sort_into_array(datasets)
    for axis, dim in enumerate(dims):
        func = partial(_remove_buffer, dim=dim)
        grid = np.apply_along_axis(func, axis, grid)

    if flat:
        return list(grid.flatten())
    else:
        return grid


def _combine_along_last_dim(datasets):
    merged = []

    # Determine the dimension along which the dataset is split
    split_dims = [d for d in datasets[0].dims if
                  len(np.unique([ds[d].values[0] for ds in datasets])) > 1]

    # Concatenate along one of the split dimensions
    concat_dim = split_dims[-1]

    # Group along the remaining dimensions and concatenate within each
    # group.
    sorted_ds = sorted(datasets, key=lambda ds: sort_key(ds, split_dims))
    for _, group in itertools.groupby(
            sorted_ds,
            key=lambda ds: tuple(ds[d].values[0] for d in split_dims[:-1])
            ):
        group = list(group)

        # Merge along concat_dim
        combined = xr.combine_nested(group, concat_dim=concat_dim)
        merged.append(combined)

    return merged


def _get_common_attrs(datasets):
    """
    Return a metadata dictionary containing all attributes that are the
    same in every dataset.

    Parameters
    ----------
    datasets : iterable of xr.Dataset or xr.DataAarray
        The datasets whose ``attrs`` properties will be compared.

    Returns
    -------
    dict
        The common metadata attributes.
    """

    attrs = {}
    not_equal = []
    for d in datasets:
        for key, val in d.attrs.items():
            if key not in attrs:
                attrs[key] = val
            elif not np.array_equal(val, attrs[key]):
                not_equal.append(key)
    attrs = {k: v for k, v in attrs.items() if k not in not_equal}
    return attrs


def auto_merge(datasets, buffer=True, chunks={}, meta_variables=[],
               use_xarray_combine=True):
    """
    Automatically merge a split xarray Dataset. This is designed to behave like
    ``xarray.open_mfdataset``, except it supports concatenation along multiple
    dimensions.

    Parameters
    ----------
    datasets : str or list of str or list of xarray.Dataset
        Either a glob expression or list of paths as you would pass to
        xarray.open_mfdataset, or a list of xarray datasets. If a list of
        datasets is passed, you should make sure that they are represented
        as dask arrays to avoid reading the whole dataset into memory.
    buffer : bool, optional
        If True, attempt to automatically remove any buffer from the tiles
        (default: True).
    meta_variables : list, optional
        A list of metadata items to concatenate as variables.
    use_xarray_combine : bool, optional
        Use xr.combine_by_coords to combine the datasets (default: True).
        Only available from ``xarray>=0.12.2``. Will fallback to a custom
        implementation if ``False`` or unavailable.

    Returns
    -------
    xarray.Dataset
        The merged dataset.

    """

    # Treat `datasets` as a glob expression
    if isinstance(datasets, str):
        datasets = glob.glob(datasets)

    if len(datasets) == 0:
        raise ValueError("No files found!")

    # Treat `datasets` as a list of file paths
    if isinstance(datasets[0], str):
        # Pass chunks={} to ensure the dataset is read as a dask array
        datasets = [add_time(xr.open_dataset(path, chunks=chunks,
                                             engine='h5netcdf'))
                    for path in datasets]

    for meta in meta_variables:
        for d in datasets:
            d[meta] = d.attrs.get(meta)

    if buffer:
        datasets = debuffer(datasets, flat=True)

    try:
        # combine_by_coords() was introduced in xarray 0.12.2
        if not use_xarray_combine:
            raise AttributeError('Requested use of custom implementation.')
        try:
            merged = xr.combine_by_coords(datasets, combine_attrs='drop')
        except TypeError:
            # For xarray < 0.16
            merged = xr.combine_by_coords(datasets)

    except AttributeError:
        merged = datasets
        while len(merged) > 1:
            merged = _combine_along_last_dim(merged)
        merged = merged[0]

    # Set metadata
    merged.attrs = _get_common_attrs(datasets)

    # Encode categorical meta variables
    for meta in meta_variables:
        # Non-numerical dtype?
        if not np.issubdtype(merged[meta].dtype, np.number):
            values, legend = merged[meta].to_series().factorize()
            merged[meta] = ('time', values.astype(int))
            merged[meta].attrs['legend'] = \
                tuple([tuple((i, v)) for i, v in enumerate(legend)])

    return merged
