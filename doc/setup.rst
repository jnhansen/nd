.. _setup:

=============
Installing nd
=============

``nd`` requires the ``libgsl-dev`` C library, so make sure you have it installed before installing ``nd``. You can find out whether it is installed by checking if the command ``gsl-config`` exists on your machine.

You may also want to install the GDAL library, but ``rasterio`` comes with a stripped down version of GDAL so for most use cases this should not be necessary.

The easiest way to install ``nd`` is via ``pip`` from the github source:

::

    pip install numpy
    pip install git+https://github.com/jnhansen/nd


Rebuilding the C extensions from Cython
---------------------------------------

In case you want to rebuild the C extensions from the
``.pyx`` files, you need to install the additional dependencies
``cython`` and ``cythongsl``. With those are installed,
``pip install`` will automatically regenerate the C files
prior to compilation.
