.. _xarray:

============================
Using ``nd`` with ``xarray``
============================

``nd`` is built on top of and around ``xarray``. As such, it is meant to be used with ``xarray.Dataset`` and ``xarray.DataArray`` objects.

Much of the functionality contained in ``nd`` is available directly from these ``xarray`` objects via custom accessors after importing the library.

Applying filters using the ``filter`` accessor
----------------------------------------------

::

    import xarray as xr
    import nd
    ds = xr.open_dataset('data/C2.nc')
    ds_filtered = ds.filter.gaussian(dims=('y', 'x'), sigma=0.5)


Reprojecting a dataset using the ``nd`` accessor
------------------------------------------------

::

    import xarray as xr
    import nd
    ds = xr.open_dataset('data/C2.nc')
    ds_proj = ds.nd.reproject(crs='EPSG:27700')
