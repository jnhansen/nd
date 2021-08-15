"""
This module provides several helper functions.
"""
import numpy as np
import xarray as xr
import multiprocess as mp
import datetime
from dateutil.tz import tzutc
from dateutil.parser import parse as parsedate
import itertools
from collections import OrderedDict
import re
from operator import add
from functools import reduce, wraps
import shutil
import importlib
import inspect


__all__ = ['get_shape',
           'get_dims',
           'str2date',
           'dict_product',
           'chunks',
           'array_chunks',
           'block_split',
           'block_merge',
           'xr_split',
           'xr_merge',
           'parallel',
           'select',
           'get_vars_for_dims',
           'expand_variables',
           'is_complex',
           'apply'
           ]


# -------------------------------------------------------------------
# Dependency checks
# -------------------------------------------------------------------
check_dependencies = {}
check_dependencies['gsl'] = (shutil.which('gsl-config') is not None)
check_dependencies['gdal'] = (shutil.which('gdal-config') is not None)
# -------------------------------------------------------------------


def check_requirements(dependency=[]):
    def _check(dep):
        if dep in check_dependencies and check_dependencies[dep]:
            return True
        try:
            importlib.import_module(dep)
        except ModuleNotFoundError:
            return False
        else:
            return True

    if isinstance(dependency, (list, tuple)):
        check = all(
            [_check(d) for d in dependency]
        )
    else:
        check = _check(dependency)

    return check


def requires(dependency=[]):
    """Class/function decorator to specify dependency requirements.

    Will raise an ImportError when the class is instantiated
    if any of the dependencies are missing. This relies on
    `nd.utils.check_dependencies`.
    """
    check = check_requirements(dependency)

    def cls_decorator(cls):
        old_init = cls.__init__

        @wraps(cls.__init__)
        def new_init(self, *args, **kwargs):
            if not check:
                raise ImportError('This class requires the following '
                                  'dependencies: {}'.format(dependency))
            return old_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls._requires = dependency
        cls._skip = not check
        return cls

    def func_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not check:
                raise ImportError('This function requires the following '
                                  'dependencies: {}'.format(dependency))
            return func(*args, **kwargs)
        return wrapper

    def decorator(obj):
        if inspect.isclass(obj):
            return cls_decorator(obj)
        else:
            return func_decorator(obj)

    return decorator


def get_shape(ds):
    # The coords are in the right order, while dims and sizes are not
    return tuple([ds.sizes[c] for c in ds.coords if c in ds.dims])


def get_dims(ds):
    """
    Return the dimension of dataset `ds` in order.
    """
    # The ordered dictionary is hidden behind two wrappers,
    # need to access the dict behind the Frozen(SortedKeysDict).
    if (xr.__version__ < '0.19.0') \
            and isinstance(ds, xr.Dataset):
        return tuple(ds.sizes.mapping.mapping)
    else:
        # The following code always works for DataArrays,
        # and also for Datasets for xarray >= 0.19.0
        return tuple(ds.sizes.mapping)


def squeeze(obj):
    """
    Return the item of an array of length 1,
    otherwise return original object.
    """
    try:
        return obj.item()
    except (ValueError, AttributeError):
        return obj


def str2date(string, fmt=None, tz=False):
    if fmt is None:
        date_object = parsedate(string)
    else:
        date_object = datetime.datetime.strptime(string, fmt)
    if tz:
        if date_object.tzinfo is None:
            date_object = date_object.replace(tzinfo=tzutc())
    elif date_object.tzinfo is not None:
        date_object = date_object.replace(tzinfo=None)
    return date_object


def dict_product(d):
    """Like itertools.product, but works with dictionaries.
    """
    return (dict(zip(d, x))
            for x in itertools.product(*d.values()))


def chunks(l, n):
    """Yield successive n-sized chunks from l.

    https://stackoverflow.com/a/312464

    Parameters
    ----------
    l : iterable
        The list or list-like object to be split into chunks.
    n : int
        The size of the chunks to be generated.

    Yields
    ------
    iterable
        Consecutive slices of l of size n.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def array_chunks(array, n, axis=0, return_indices=False):
    """Chunk an array along the given axis.

    Parameters
    ----------
    array : numpy.array
        The array to be chunked
    n : int
        The chunksize.
    axis : int, optional
        The axis along which to split the array into chunks (default: 0).
    return_indices : bool, optional
        If True, yield the array index that will return chunk rather
        than the chunk itself (default: False).

    Yields
    ------
    iterable
        Consecutive slices of `array` of size `n`.
    """
    if axis >= array.ndim:
        raise ValueError(
            "axis {:d} is out of range for given array."
            .format(axis)
        )

    arr_len = array.shape[axis]
    for i in range(0, arr_len, n):
        indices = [slice(None), ] * array.ndim
        indices[axis] = slice(i, i+n)
        if return_indices:
            yield indices, array[tuple(indices)]
        else:
            yield array[tuple(indices)]


def block_split(array, blocks):
    """Split an ndarray into subarrays according to blocks.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be split.
    blocks : array_like
        The desired number of blocks per axis.

    Returns
    -------
    list
        A list of blocks, in column-major order.

    Examples
    --------
    >>> block_split(np.arange(16).reshape((4, 4)), (2, 2))
    [array([[ 0,  1],
            [ 4,  5]]),
     array([[ 2,  3],
            [ 6,  7]]),
     array([[ 8,  9],
            [12, 13]]),
     array([[10, 11],
            [14, 15]])]
    """
    if array.ndim != len(blocks):
        raise ValueError(
            "Length of 'blocks' must be equal to the "
            "array dimensionality."
        )

    result = [array]
    for axis, nblocks in enumerate(blocks):
        result = [np.array_split(_, nblocks, axis=axis) for _ in result]
        result = [item for sublist in result for item in sublist]
    return result


def block_merge(array_list, blocks):
    """Reassemble a list of arrays as generated by block_split.

    Parameters
    ----------
    array_list : list of numpy.array
        A list of numpy.array, e.g. as generated by block_split().
    blocks : array_like
        The number of blocks per axis to be merged.

    Returns
    -------
    numpy.array
        A numpy array with dimension len(blocks).
    """
    if len(array_list) != np.prod(blocks):
        raise ValueError(
            "Length of array list must be equal to the "
            "product of the shape elements."
        )

    result = array_list
    for i, nblocks in enumerate(blocks[::-1]):
        axis = len(blocks) - i - 1
        result = [np.concatenate(_, axis=axis)
                  for _ in chunks(result, nblocks)]
    return result[0]


def xr_split(ds, dim, chunks, buffer=0):
    """Split an xarray Dataset into chunks.

    Parameters
    ----------
    ds : xarray.Dataset
        The original dataset
    dim : str
        The dimension along which to split.
    chunks : int
        The number of chunks to generate.

    Yields
    ------
    xarray.Dataset
        An individual chunk.
    """
    n = ds.sizes[dim]
    chunksize = int(np.ceil(n / chunks))
    for i in range(chunks):
        low = max(i * chunksize - buffer, 0)
        high = min((i+1) * chunksize + buffer, n)
        idx = slice(low, high)
        chunk = ds.isel(**{dim: idx})
        yield chunk


def xr_merge(ds_list, dim, buffer=0):
    """Reverse xr_split().

    Parameters
    ----------
    ds_list : list of xarray.Dataset
    dim : str
        The dimension along which to concatenate.

    Returns
    -------
    xarray.Dataset
    """
    if buffer > 0 and len(ds_list) > 1:
        # Explicit conversion to int fixes
        # underflow issue that sometimes arises
        # when called from @algorithm.parallelize decorated function
        idx_first = slice(None, -int(buffer))
        idx_middle = slice(buffer, -int(buffer))
        idx_end = slice(buffer, None)
        parts = [ds_list[0].isel(**{dim: idx_first})] + \
                [ds.isel(**{dim: idx_middle}) for ds in ds_list[1:-1]] + \
                [ds_list[-1].isel(**{dim: idx_end})]
    else:
        parts = ds_list
    return xr.concat(parts, dim=dim)


def parallel(fn, dim=None, chunks=None, chunksize=None, merge=True, buffer=0):
    """
    Parallelize a function that takes an xarray dataset as first argument.

    TODO: make accept numpy arrays as well.

    Parameters
    ----------
    fn : function
        *Must* take an xarray.Dataset as first argument.
    dim : str, optional
        The dimension along which to split the dataset for parallel execution.
        If not passed, try 'y' as default dimension.
    chunks : int, optional
        The number of chunks to execute in parallel. If not passed, use the
        number of available CPUs.
    chunksize : int, optional
        ... to be implemented
    buffer : int, optional
        (default: 0)

    Returns
    -------
    function
        A parallelized function that may be called with exactly the same
        arguments as `fn`.
    """
    if dim is None:
        dim = 'y'
    if chunks is None:
        chunks = mp.cpu_count()

    def wrapper(ds, *args, **kwargs):
        if dim not in ds.dims:
            raise ValueError(
                "The dataset has no dimension '{}'."
                .format(dim)
            )

        # Split into parts
        parts = xr_split(
            ds, dim=dim, chunks=chunks, buffer=buffer)

        def _fn(ds):
            return fn(ds, *args, **kwargs)

        pool = mp.Pool(chunks)
        output = pool.map(_fn, parts)
        pool.close()
        pool.join()

        if merge:
            result = xr_merge(output, dim=dim, buffer=buffer)
        else:
            result = output

        return result

    return wrapper


def select(objects, fn, unlist=True, first=False):
    """Returns a subset of `objects` that matches a range of criteria.

    Parameters
    ----------
    objects : list of obj
        The collection of objects to filter.
    fn : lambda expression
        Filter objects by whether fn(obj) returns True.
    first: bool, optional
        If True, return first entry only (default: False).
    unlist : bool, optional
        If True and the result has length 1 and objects is a list, return the
        object directly, rather than the list (default: True).

    Returns
    -------
    list
        A list of all items in `objects` that match the specified criteria.

    Examples
    --------
    >>> select([{'a': 1, 'b': 2}, {'a': 2, 'b': 2}, {'a': 1, 'b': 1}],
                lambda o: o['a'] == 1)
    [{'a': 1, 'b': 2}, {'a': 1, 'b': 1}]
    """
    filtered = objects
    if type(objects) is list:
        filtered = [obj for obj in filtered if fn(obj)]
    elif type(objects) is dict:
        filtered = {obj_key: obj for obj_key, obj
                    in filtered.items() if fn(obj)}
    if first:
        if len(filtered) == 0:
            return None
        elif type(filtered) is list:
            return filtered[0]
        elif type(filtered) is dict:
            return filtered[list(filtered.keys())[0]]
    elif unlist and len(filtered) == 1 and \
            type(filtered) is list:
        return filtered[0]
    else:
        return filtered


def get_vars_for_dims(ds, dims, invert=False):
    """
    Return a list of all variables in `ds` which have dimensions `dims`.

    Parameters
    ----------
    ds : xarray.Dataset
    dims : list of str
        The dimensions that each variable must contain.
    invert : bool, optional
        Whether to return the variables that do *not* contain the given
        dimensions (default: False).

    Returns
    -------
    list of str
        A list of all variable names that have dimensions `dims`.
    """
    return [v for v in ds.data_vars
            if set(ds[v].dims).issuperset(set(dims)) != invert]


def expand_variables(da, dim='variable'):
    """
    This is the inverse of xarray.Dataset.to_array().

    Parameters
    ----------
    da : xarray.DataArray
        A DataArray that contains the variable names as dimension.
    dim : str
        The dimension name (default: 'variable').

    Returns
    -------
    xarray.Dataset
        A dataset with the variable dimension in `da` exploded to variables.
    """
    _vars = []
    attrs = da.attrs
    da.attrs = {}
    for v in da[dim]:
        _var = da.sel(**{dim: v})
        _var.name = str(_var[dim].values)
        del _var[dim]
        _vars.append(_var)

    result = xr.merge(_vars)
    result.attrs = attrs
    return result


def is_complex(ds):
    """Check if a dataset contains any complex variables.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    bool
        True if `ds` contains any complex variables, False otherwise.
    """
    if isinstance(ds, xr.DataArray):
        return np.iscomplexobj(ds)
    elif isinstance(ds, xr.Dataset):
        return np.any(
            [np.iscomplexobj(v) for v in ds.data_vars.values()]
        )
    else:
        raise ValueError(
            "Not an xarray Dataset or DataArray: {}"
            .format(repr(ds))
        )


def _wlen(s):
    """Return number of leading whitespace characters."""
    return len(s) - len(s.lstrip())


def parse_docstring(doc):
    parsed = OrderedDict()

    if doc is None:
        return parsed

    lines = doc.split('\n')

    # Find indentation level and reset to 0
    # Exclude first and last line
    if len(lines) < 3:
        indent = 0
    else:
        indent = min([_wlen(_) for _ in lines[1:-1] if len(_.lstrip()) > 0])
    lines = [l[indent:] if _wlen(l) >= indent else l for l in lines]
    parsed['indent'] = indent

    # Find sections
    line_numbers = np.arange(len(lines))
    rule = re.compile('^ *-+$')
    section_starts = list(line_numbers[np.array([
        rule.match(l) is not None for l in lines])])

    if len(section_starts) == 0:
        parsed[None] = lines
    else:
        parsed[None] = lines[:section_starts[0] - 2]

        # Iterate through sections
        for start, stop in zip(section_starts, section_starts[1:] + [None]):
            section_name = lines[start - 1].strip()
            if stop is not None:
                stop -= 2
            section = lines[start + 1:stop]

            # Split section contents by parameter
            param_starts = [i for i, s in enumerate(section) if _wlen(s) == 0]
            parsed[section_name] = \
                [section[pstart:pstop] for pstart, pstop in
                 zip(param_starts, param_starts[1:]+[None])]

    return parsed


def assemble_docstring(parsed, sig=None):
    """
    Assemble a docstring from an OrderedDict as returned by
    :meth:`nd.utils.parse_docstring()`

    Parameters
    ----------
    parsed : OrderedDict
        A parsed docstring as obtained by ``nd.utils.parse_docstring()``.
    sig : function signature, optional
        If provided, the parameters in the docstring will be ordered according
        to the parameter order in the function signature.

    Returns
    -------
    str
        The assembled docstring.

    """

    parsed = parsed.copy()
    indent = parsed.pop('indent')
    pad = ' '*indent

    # Sort 'Parameters' section according to signature
    if sig is not None and 'Parameters' in parsed:
        order = tuple(sig.parameters.keys())

        def sort_index(p):
            key = p[0].split(':')[0].strip(' *')
            if key == '':
                return 9999
            return order.index(key)

        parsed['Parameters'] = sorted(parsed['Parameters'], key=sort_index)

    d = []
    for k, v in parsed.items():
        if isinstance(v[0], list):
            flat_v = reduce(add, v)
        else:
            flat_v = v

        if k is not None:
            d.extend(['', pad + k, pad + '-'*len(k)])

        d.extend([(pad + l).rstrip() for l in flat_v])

    return '\n'.join(d)


def apply(ds, fn, signature=None, njobs=1):
    """
    Apply a function to a Dataset that operates on a defined
    subset of dimensions.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The dataset to which to apply the function.
    fn : function
        The function to apply to the Dataset.
    signature : str, optional
        The signature of the function in dimension names,
        e.g. '(time,var)->(time)'.
        If 'var' is included, the Dataset variables will be converted into
        a new dimension and the result will be a DataArray.
    njobs : int, optional
        The number of jobs to run in parallel.

    Returns
    -------
    xr.Dataset
        The output dataset with changed dimensions according to the
        function signature.
    """

    def _parse_signature(sig):
        if sig is None:
            sig = '(time,var)->(time)'
        m = re.match('\((.*)\)->\((.*)\)', sig)
        dims = tuple(group.split(',') if len(group) > 0 else []
                     for group in m.groups())
        if len(dims) != 2:
            raise ValueError(
                "Invalid signature: Signature must be of the "
                "form '(<dim1>,...,<dimN>)->(<dim1>,...,<dimM>)'."
            )
        return dims

    dims_in, dims_out = _parse_signature(signature)

    # All output dimensions must also be input dimensions
    if len(dims_out) > 0 and not set(dims_out).issubset(dims_in):
        raise ValueError(
            "Invalid signature: All output dimensions must "
            "also be input dimensions."
        )

    # Vectorize function
    fn_vec = np.vectorize(fn, signature=signature)

    # If 'var' is an input, convert variables to new dimension.
    if isinstance(ds, xr.Dataset) and 'var' in dims_in:
        ds = ds.to_array(dim='var')

    # Determine all extra dimensions
    src_dims = get_dims(ds)
    dims_removed = set(dims_in) - set(dims_out)
    output_dims = [d for d in src_dims if d not in dims_removed]
    extra_dims = tuple(set(src_dims) - set(dims_in))

    # Flatten extra dimensions
    ds = ds.stack(z=extra_dims).transpose('z', *dims_in)

    def apply_func(ds):
        # Apply function and turn into DataArray
        result_arr = fn_vec(ds)
        dims = ('z',) + tuple(dims_out)
        result = xr.DataArray(
            result_arr,
            dims=dims, coords={d: ds.coords[d] for d in dims}
        )
        return result

    # Parallelize
    if njobs != 1:
        chunks = njobs if njobs > 0 else None
        apply_func = parallel(apply_func, dim='z', chunks=chunks)

    # Apply function
    if isinstance(ds, xr.DataArray):
        result = apply_func(ds)
    elif isinstance(ds, xr.Dataset):
        # Apply to each variable as DataArray
        try:
            result = ds.map(apply_func)
        except AttributeError:
            # Backwards compatibility for xarray < 0.14.1:
            result = ds.apply(apply_func)

    # Restore original dimension order
    result = result.unstack().transpose(*output_dims)

    # Turn back into Dataset
    if 'var' in result.dims:
        result = result.to_dataset(dim='var')

    return result


def extract_arguments(fn, args, kwargs):
    """
    Given a function fn, return the leftover `*args` and `**kwargs`.
    """
    def _(*args, **kwargs):
        pass
    sig = inspect.signature(fn)

    # Remove 'self' parameter
    if 'self' in sig.parameters:
        sig = sig.replace(parameters=tuple(sig.parameters.values())[1:])

    # Use an OrderedDict to maintain the parameter order in the signature
    parameters = OrderedDict(sig.parameters)
    parameters.update(OrderedDict(inspect.signature(_).parameters))
    parameters = sorted(
        parameters.values(),
        key=lambda p: (p.kind, p.default is not inspect._empty)
    )
    new_sig = sig.replace(parameters=parameters)
    bound = new_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments
