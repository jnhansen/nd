.. _nlmeans:

===============
Non-Local Means
===============

Non-Local Means is a denoising filter that computes filtered pixel values as a weighted average of pixels in the spatial neighborhood, where the weights are determined as a function of color distance.

Example::

   from geo.filters import nlmeans
   ds_filtered = nlmeans(ds, r={'lat': 3, 'lon': 3, 'time': 1},
                         sigma=1, h=1, f=1)


.. topic:: See Also:

 * :meth:`nd.filters.nlmeans`


.. topic:: References:

 * Buades, A., Coll, B., & Morel, J.-M. (2011).
   `Non-Local Means Denoising <https://doi.org/10.5201/ipol.2011.bcm_nlm>`_.
   Image Processing On Line, 1, 208–212.
