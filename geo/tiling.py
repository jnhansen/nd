"""
This module may be used to mosaic and tile multiple satellite image products.

TODO: Contain buffer information in NetCDF metadata?

"""
from .io import from_netcdf, to_netcdf, add_time
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
    path : str
    prefix : str, optional
    chunks : dict, optional
    buffer : int or dict, optional
        The number of overlapping pixels to store around each tile
        (default: 0).
    """
    # Prepare output directory
    if os.path.isfile(path):
        raise ValueError("`path` cannot be a file!")
    elif not os.path.isdir(path):
        os.makedirs(path)

    #  Treat `ds` as a file path
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)

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
        elif isinstance(buffer, dict):
            _buf = buffer[dim] if dim in buffer else 0
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

    def _write_tile(slices):
        # Slice the dataset and write to disk.
        subset = ds.isel(slice_dict)
        suffix = '.'.join(
            ['{}_{}'.format(dim, s.start) for dim, s in slice_dict.items()]
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
    """
    if isinstance(files, str):
        files = glob.glob(files)

    if path is not None:
        os.makedirs(path, exist_ok=True)

    def _wrapper(f):
        # 1. Open dataset
        data = from_netcdf(f)
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
        * If 'auto' (default), attempt to automatically detect the buffer for
        each dimension.
        * If int, it is the number of overlapping pixels stored around each
        tile
        * If dict, this is the amount of buffer per dimension

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
        datasets = [add_time(xr.open_dataset(path, chunks=chunks))
                    for path in datasets]

    if buffer == 'auto':
        buf_cache = {}

    def _combine_along_last_dim(datasets):
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
                else:
                    # descending order
                    keys.append(-vals[0])
            return tuple(keys)

        sorted_ds = sorted(datasets, key=lambda ds: sort_key(ds, split_dims))
        for _, group in itertools.groupby(
                sorted_ds,
                key=lambda ds: tuple(ds[d].values[0] for d in split_dims[:-1])
                ):
            group = list(group)

            # Auto-detect buffer?
            if buffer == 'auto':
                if concat_dim in buf_cache:
                    _buf = buf_cache[concat_dim]
                else:
                    # Determine overlap along concat_dim
                    values0 = group[0][concat_dim].values
                    values1 = group[1][concat_dim].values
                    diff = values1 - values0[-1]
                    _buf = int((np.argmin(np.abs(diff)) + 1) / 2)
                    buf_cache[concat_dim] = _buf

            elif isinstance(buffer, int):
                _buf = buffer

            elif isinstance(buffer, dict):
                _buf = buffer[concat_dim] if concat_dim in buffer else 0

            else:
                _buf = 0

            # Apply buffer
            if _buf > 0:
                idx_first = {concat_dim: slice(None, -_buf)}
                idx_middle = {concat_dim: slice(_buf, -_buf)}
                idx_end = {concat_dim: slice(_buf, None)}
                # Requires that group is sorted by concat_dim
                group = [group[0].isel(idx_first)] + \
                        [_.isel(idx_middle) for _ in group[1:-1]] + \
                        [group[-1].isel(idx_end)]

            # Merge along concat_dim
            merged.append(xr.auto_combine(group, concat_dim=concat_dim))

        return merged

    merged = datasets
    while len(merged) > 1:
        merged = _combine_along_last_dim(merged)

    # Close opened files
    for d in datasets:
        d.close()

    return merged[0]


if __name__ == '__main__':
    pass
