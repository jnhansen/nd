from ..algorithm import Algorithm
from abc import abstractmethod
from ..utils import get_vars_for_dims, expand_variables, is_complex
from ..io import disassemble_complex, assemble_complex


class Filter(Algorithm):
    """
    The base class for a generic filter.

    Parameters
    ----------
    dims : tuple of str
        The dimensions along which the filter is applied.
    """

    # If per_variable is True, the filter is applied independently for
    # each variable. Otherwise, all variables may be used to determine the
    # filter weights.
    per_variable = True
    # If supports_complex is False, complex-values variables are disassembled
    # into two reals before applying the filter and reassembled afterwards.
    supports_complex = False
    dims = ()

    @abstractmethod
    def __init__(self, *args, **kwargs):
        return

    def apply(self, ds, inplace=False):
        """
        Apply the filter to the input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset
        inplace : bool, optional
            If True, overwrite the input data inplace (default: False).

        Returns
        -------
        xarray.Dataset
            The filtered dataset
        """
        if inplace:
            raise NotImplementedError('Inplace filtering is not currently '
                                      'implemented.')

        # This is not in the correct order, as ds.dims is always sorted
        # alphabetically.
        orig_dims = tuple(ds.dims)
        ordered_dims = self.dims + tuple(set(orig_dims) - set(self.dims))

        # Check if any of the variables are complex
        convert_complex = is_complex(ds) and not self.supports_complex
        if convert_complex:
            disassemble_complex(ds, inplace=True)

        # Find all variables that match the given dimensions
        variables = get_vars_for_dims(ds, self.dims)
        other_variables = get_vars_for_dims(ds, self.dims, invert=True)

        #
        # Apply the actual filter
        #
        if self.per_variable:
            # Apply independently for each variable.
            result = ds.copy(deep=True)
            for v in variables:
                vdims = result[v].dims
                axes = tuple([vdims.index(d) for d in self.dims])
                # Prepare data and output as numpy arrays
                self._filter(ds[v].values, axes,
                             output=result[v].values)

        else:
            # The variables are an additional dimension.
            ordered_dims = ordered_dims + ('variable',)

            # convert to DataArray
            da_ordered = ds[variables].to_array().transpose(*ordered_dims)
            da_filtered = da_ordered.copy(deep=True)
            axes = tuple([da_ordered.dims.index(d) for d in self.dims])

            # Prepare data and output as numpy arrays
            self._filter(da_ordered.values, axes, output=da_filtered.values)

            # Reassemble Dataset
            result = expand_variables(da_filtered)
            # Make sure all variable dimensions are in the original order
            for v in result.data_vars:
                result[v] = result[v].transpose(*ds[v].dims)

            for v in other_variables:
                result[v] = ds[v]

        # Reassemble complex variabbles if previously disassembled
        if convert_complex:
            assemble_complex(ds, inplace=True)

        return result

    @abstractmethod
    def _filter(self, arr, axes, output=None):
        """
        This method must be implemented by all derived classes.
        """
        return
