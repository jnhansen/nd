.. _convolutions:

===================
Kernel Convolutions
===================

:mod:`nd.filters` implements generic kernel convolutions as well as several convenience functions
for special cases. Every filter supports the argument ``dims`` to specify a subset of dimensions along which to apply the filter. If a kernel is given, the number of dimensions must match the shape of the kernel.

The following example performs a simple Sobel edge detection filter along the ``lon`` dimension using :meth:`nd.filters.convolve`.

Example::

   from nd.filters import convolve
   kernel = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
   edges = convolve(ds, kernel, dims=('lat', 'lon'))

A boxcar convolution (see :meth:`nd.filters.boxcar`) uses a square kernel (in `n` dimensions) with equal weights for each pixel. The total weight is normalized to one.

Example::

   from nd.filters import boxcar
   smooth = boxcar(ds, w=3, dims=('lat', 'lon'))


A Gaussian filter (see :meth:`nd.filters.gaussian`) convolves the datacube with a Gaussian kernel.

Example::

   from nd.filters import gaussian
   smooth = gaussian(ds, sigma=1, dims=('lat', 'lon'))
