"""
.. document private functions
.. autofunction:: _cluster
"""

from .classification_ import (cluster, norm_by_cluster, _cluster,
                              Classifier, rasterize)

__all__ = ['_cluster', 'cluster', 'norm_by_cluster',
           'Classifier', 'rasterize']
