.. _setup:

=============
Installing nd
=============

``nd`` requires the ``gdal`` and ``libgsl-dev`` C libraries, so make sure you have those installed before installing ``nd``. You can find out whether they are installed by checking if the commands ``gdal-config`` and ``gsl-config`` exist on your machine.

The easiest way to install ``nd`` is via ``pip`` from the github source:

::

    pip install numpy cythongsl
    pip install git+https://github.com/jnhansen/nd


