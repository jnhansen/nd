from ..algorithm import Algorithm


class ChangeDetection(Algorithm):

    njobs = 1

    def __init__(self, njobs=1):
        self.njobs = njobs
