# Strangely, importing h5netcdf must be imported at this point
# for the h5netcdf backend to work properly on Travis (and Ubuntu)
try:
    import h5netcdf
except Exception:
    pass

__all__ = ['change', 'classify', 'io', 'visualize', 'filters', 'utils']
