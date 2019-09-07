"""
This module may be used to mosaic and tile multiple satellite image products.

TODO: Contain buffer information in NetCDF metadata?

"""
from .io import open_netcdf, to_netcdf, add_time
from . import utils
import os
import glob
import itertools
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


def _combined_attrs(datasets):
    attrs = {}
    for ds in datasets:
        attrs.update(ds.attrs)
    return attrs


def _combine_along_last_dim(datasets, buffer):
    merged = []

    # Determine the dimension along which the dataset is split
    split_dims = [d for d in datasets[0].dims if
                  len(np.unique([ds[d].values[0] for ds in datasets])) > 1]

    # Concatenate along one of the split dimensions
    concat_dim = split_dims[-1]

    # Group along the remaining dimensions and concatenate within each
    # group.
    def sort_key(ds, dims):
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

    sorted_ds = sorted(datasets, key=lambda ds: sort_key(ds, split_dims))
    for _, group in itertools.groupby(
            sorted_ds,
            key=lambda ds: tuple(ds[d].values[0] for d in split_dims[:-1])
            ):
        group = list(group)

        #
        # Compute slices based on the buffer
        #

        # Auto-detect buffer?
        if buffer == 'auto':
            values = [d[concat_dim].values for d in group]

            # overlap with previous
            prev_diffs = [np.abs(values[i] - values[i-1][-1])
                          for i in range(1, len(values))]

            # The +1 is necessary because this is the stop index of a slice
            # cast to float because it may be a timedelta
            prev_idx = [int(np.argmin(diff)) + 1
                        if float(np.min(diff)) == 0 else 0
                        for diff in prev_diffs]
            prev_idx = [None] + [int(np.ceil(i/2)) for i in prev_idx]

            # overlap with next
            next_diffs = [np.abs(values[i] - values[i+1][0])
                          for i in range(0, len(values)-1)]
            next_idx = [int(np.argmin(diff))
                        for diff in next_diffs]

            next_idx = [int(np.ceil((i + len(v)) / 2))
                        for i, v in zip(next_idx, values)] + [None]

            slices = [slice(*args) for args in zip(prev_idx, next_idx)]

        else:
            if isinstance(buffer, int):
                _buf = buffer

            elif isinstance(buffer, dict):
                _buf = buffer[concat_dim] if concat_dim in buffer else 0

            else:
                _buf = 0

            slices = [slice(None, -_buf)] + \
                     [slice(_buf, -_buf)] * (len(group) - 2) + \
                     [slice(_buf, None)]

        # Apply buffered slices
        idx = [{concat_dim: s} for s in slices]
        group = [d.isel(i) for d, i in zip(group, idx)]

        # Merge along concat_dim
        combined = xr.combine_nested(group, concat_dim=concat_dim)
        combined.attrs = _combined_attrs(group)
        merged.append(combined)

    return merged


def auto_merge(datasets, buffer='auto', chunks={}):
    """
    Automatically merge a split xarray Dataset. This is designed to behave like
    `xarray.open_mfdataset`, except it supports concatenation along multiple
    dimensions.

    Parameters
    ----------
    datasets : str or list of str or list of xarray.Dataset
        Either a glob expression or list of paths as you would pass to
        xarray.open_mfdataset, or a list of xarray datasets. If a list of
        datasets is passed, you should make sure that they are represented
        as dask arrays to avoid reading the whole dataset into memory.
    buffer : 'auto' or int or dict, optional

        - If 'auto' (default), attempt to automatically detect the buffer for \
        each dimension.

        - If `int`, it is the number of overlapping pixels stored around each \
        tile

        - If `dict`, this is the amount of buffer per dimension

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

    if buffer == 'auto':
        buf_cache = {}

    merged = datasets
    while len(merged) > 1:
        merged = _combine_along_last_dim(merged, buffer)

    # Close opened files
    for d in datasets:
        d.close()

    return merged[0]
