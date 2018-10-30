from ..algorithm import Algorithm
import os


class ChangeDetection(Algorithm):

    njobs = 1

    def __init__(self, njobs=1):
        self.njobs = njobs
        os.environ['OMP_NUM_THREADS'] = str(njobs)
