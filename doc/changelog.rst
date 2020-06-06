Changelog
=========

Version 0.2 (*under development*)
-----------------------------------

General updates
...............

- add support for Python 3.8

:mod:`nd.classify`
..................

- removed redundant method :func:`nd.classify.cluster()`, as same
  functionality can be achieved using :class:`nd.classify.Classifier`

:mod:`nd.tiling`
................

- added :func:`nd.tiling.debuffer()` to automatically remove buffer from
  tiled datasets

:mod:`nd.visualize`
...................

- added :func:`nd.visualize.plot_map()` to plot the geometry of a dataset
  on a map

- added :func:`nd.visualize.gridlines_with_labels()` to add perfectly aligned
  tick labels around a map with gridlines

