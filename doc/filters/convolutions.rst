.. _convolutions:

===================
Kernel Convolutions
===================

:mod:`nd.filters` implements generic kernel convolutions as well as several convenience functions
for special cases. Every filter supports the argument ``dims`` to specify a subset of dimensions along which to apply the filter. If a kernel is given, the number of dimensions must match the shape of the kernel.

The following example performs a simple Sobel edge detection filter along the ``x`` dimension using :meth:`nd.filters.ConvolutionFilter`.

Example::

   from nd.filters import ConvolutionFilter
   kernel = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
   edges = ConvolutionFilter(kernel, dims=('y', 'x'))
   edges = conv.apply(ds)

A boxcar convolution (see :meth:`nd.filters.BoxcarFilter`) uses a square kernel (in `n` dimensions) with equal weights for each pixel. The total weight is normalized to one.

Example::

   from nd.filters import BoxcarFilter
   boxcar = BoxcarFilter(w=3, dims=('y', 'x'))
   smooth = boxcar.apply(ds)


A Gaussian filter (see :meth:`nd.filters.GaussianFilter`) convolves the datacube with a Gaussian kernel.

Example::

   from nd.filters import GaussianFilter
   gaussian = GaussianFilter(sigma=1, dims=('y', 'x'))
   smooth = gaussian.apply(ds)
