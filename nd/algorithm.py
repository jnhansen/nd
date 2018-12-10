"""
This module contains the abstract base class Algorithm.
"""
from abc import ABC, abstractmethod
from .utils import parallel


class Algorithm(ABC):

    @abstractmethod
    def apply(self, ds):
        return

    def parallel_apply(self, ds, dim, jobs=None):
        return parallel(self.apply, dim=dim, chunks=jobs)
