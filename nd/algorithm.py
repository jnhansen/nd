"""
This module contains the abstract base class Algorithm.
"""
from abc import ABC, abstractmethod, ABCMeta
import inspect
from types import CodeType
import sys
import multiprocessing as mp
from functools import partial
from . import utils

PY_VERSION = sys.version_info


class Algorithm(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, ds):
        """
        Must be implemented by derived classes and should be given
        @parallelize decorator where appropriate.
        """
        return

    def _buffer(self, dim):
        """
        Return the required tile buffer when parallelizing over the given
        dimension.
        """
        return 0

    def _parallel_dimension(self, ds):
        """Return the dimension along which to parallelize"""
        return 'y'


def parallelize(func):
    """
    Decorator. Parallelize a function that takes an xarray dataset as first
    argument.

    Parameters
    ----------
    fn : function
        *Must* take an xarray.Dataset as first argument.

    Returns
    -------
    function
        A parallelized function that may be called with exactly the same
        arguments as `fn`.
    """

    # Create wrapper function
    # -----------------------
    def wrapper(self, ds, *args, njobs=1, **kwargs):
        method = partial(func, self)
        if njobs == -1:
            njobs = mp.cpu_count()
        if njobs == 1:
            return method(ds, *args, **kwargs)
        else:
            dim = self._parallel_dimension(ds)
            buffer = self._buffer(dim)
            return utils.parallel(
                method, dim=dim, chunks=njobs, buffer=buffer
            )(ds, *args, **kwargs)

    # Override signature
    # ------------------
    sig_func = inspect.signature(func)
    sig_wrapper = inspect.signature(wrapper)
    parameters = tuple(sig_func.parameters.values())
    parameters += (sig_wrapper.parameters['njobs'],)
    # Sort parameters:
    # 1) put variadic parameters last
    # 2) put arguments without defaults first
    parameters = sorted(
        parameters,
        key=lambda p: (p.kind, p.default is not inspect._empty)
    )
    new_parameters = []
    for p in parameters:
        if p not in new_parameters:
            new_parameters.append(p)
    sig = sig_func.replace(parameters=new_parameters)

    # Override docstring
    # ------------------
    doc = utils.parse_docstring(func.__doc__)
    if 'Parameters' not in doc:
        doc['Parameters'] = []
    doc['Parameters'].append(
        ['njobs : int, optional',
         '    Number of jobs to run in parallel. Setting njobs to -1 ',
         '    uses the number of available cores.',
         '    Disable parallelism by setting njobs to 1 (default).']
    )
    doc = utils.assemble_docstring(doc, sig=sig)

    wrapper.__signature__ = sig
    wrapper.__doc__ = doc

    return wrapper


def wrap_algorithm(algo, name=None):
    """
    Return the function representation of an Algorithm derived class.

    NOTE: If algo.apply has ``*args`` and ``**kwargs`` parameters,
    this doesn't work.
    """
    if not issubclass(algo, Algorithm):
        raise ValueError('Class must be an instance of `nd.Algorithm`.')

    def _wrapper(*args, **kwargs):
        # First, apply the arguments to .apply():
        apply_kwargs = utils.extract_arguments(algo.apply, args, kwargs)
        init_args = apply_kwargs.pop('args', ())
        init_kwargs = apply_kwargs.pop('kwargs', {})
        return algo(*init_args, **init_kwargs).apply(**apply_kwargs)

    # Override function module
    _wrapper.__module__ = algo.__module__

    # Override position of source code to fix source code links in
    # documentation. This is probably a bad idea.
    code = _wrapper.__code__
    new_filename = inspect.getfile(algo)
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    new_firstlineno = caller.lineno
    if PY_VERSION >= (3, 8):
        new_code = code.replace(
            co_filename=new_filename,
            co_firstlineno=new_firstlineno
        )
    else:
        new_code = CodeType(
            code.co_argcount,
            code.co_kwonlyargcount,
            code.co_nlocals,
            code.co_stacksize,
            code.co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
            new_filename,
            code.co_name,
            new_firstlineno,
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars
        )
    _wrapper.__code__ = new_code

    # Override function name
    if name is not None:
        _wrapper.__name__ = name

    # Override signature
    sig_init = inspect.signature(algo.__init__)
    sig_apply = inspect.signature(algo.apply)
    parameters = tuple(sig_apply.parameters.values())[1:] + \
        tuple(sig_init.parameters.values())[1:]
    # Sort parameters:
    # 1) put variadic parameters last
    # 2) put arguments without defaults first
    parameters = sorted(
        parameters,
        key=lambda p: (p.kind, p.default is not inspect._empty)
    )
    new_parameters = []
    for p in parameters:
        if p not in new_parameters:
            new_parameters.append(p)
    sig = sig_init.replace(parameters=new_parameters)
    _wrapper.__signature__ = sig

    # Override docstring
    link = ':class:`{}.{}`'.format(algo.__module__, algo.__name__)
    doc = utils.parse_docstring(algo.__doc__)
    doc[None].insert(0, "Wrapper for {}.".format(link))
    doc[None].insert(1, "")
    if algo.apply.__doc__ is not None:
        apply_doc = utils.parse_docstring(algo.apply.__doc__)
        if 'Parameters' in apply_doc:
            if 'Parameters' not in doc:
                doc['Parameters'] = apply_doc['Parameters']
            else:
                doc['Parameters'] = apply_doc['Parameters'] + doc['Parameters']
        if 'Returns' in apply_doc:
            doc['Returns'] = apply_doc['Returns']
    _wrapper.__doc__ = utils.assemble_docstring(doc, sig=sig)

    return _wrapper
