"""
This module contains the abstract base class Algorithm.
"""
from abc import ABC, abstractmethod, ABCMeta
import inspect
from types import CodeType
from collections import OrderedDict
from . import utils


class Algorithm(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, ds):
        return

    def parallel_apply(self, ds, dim, jobs=None):
        return utils.parallel(self.apply, dim=dim, chunks=jobs)


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
    new_sig = sig.replace(parameters=tuple(parameters.values()))
    bound = new_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


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
        apply_kwargs = extract_arguments(algo.apply, args, kwargs)
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

    # Override docstring
    link = ':class:`{}.{}`'.format(algo.__module__, algo.__name__)
    doc = utils.parse_docstring(algo.__doc__)
    doc[None].insert(0, "Wrapper for {}.".format(link))
    doc[None].insert(1, "")
    if algo.apply.__doc__ is not None:
        apply_doc = utils.parse_docstring(algo.apply.__doc__)
        if 'Parameters' in apply_doc:
            doc['Parameters'] = apply_doc['Parameters'] + doc['Parameters']
        if 'Returns' in apply_doc:
            doc['Returns'] = apply_doc['Returns']
    _wrapper.__doc__ = utils.assemble_docstring(doc)

    # Override signature
    sig_init = inspect.signature(algo.__init__)
    sig_apply = inspect.signature(algo.apply)
    parameters = tuple(sig_apply.parameters.values())[1:] + \
        tuple(sig_init.parameters.values())[1:]
    sig = sig_init.replace(parameters=parameters)
    _wrapper.__signature__ = sig

    return _wrapper
