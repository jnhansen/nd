# Strangely, importing h5netcdf must be imported at this point
# for the h5netcdf backend to work properly on Travis (and Ubuntu)
import h5netcdf

__all__ = ['change', 'classify', 'io', 'visualize', 'filters', 'utils']
