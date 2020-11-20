.. _setup:

=============
Installing nd
=============

You may also want to install the GDAL library, but ``rasterio`` comes with a stripped down version of GDAL so for most use cases this should not be necessary.

The easiest way to install ``nd`` is via ``pip`` from PyPI::

    pip install numpy
    pip install nd

You can also install the latest version from Github::

    pip install git+https://github.com/jnhansen/nd


Some algorithms require the ``libgsl-dev`` C library:

- ``nd.change.OmnibusTest``

If you want to use these algorithms you need to make sure you have the library installed *before* installing ``nd``. You can find out whether it is installed by checking if the command ``gsl-config`` exists on your machine.

Rebuilding the C extensions from Cython
---------------------------------------

In case you want to rebuild the C extensions from the
``.pyx`` files, you need to install the additional dependencies
``cython`` and ``cythongsl``. With those installed,
``pip install`` will automatically regenerate the C files
prior to compilation.
