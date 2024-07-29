# Standard
import random
import time

# Local
from ...config import FMArguments


class MockSpeedEstimator:
    def __init__(self, fm_args: FMArguments, seed=None):
        self.fm = fm_args

        if seed is not None:
            self.seed = seed
        else:
            self.seed = time.time()

    def get_tps(self, seqlen=None):
        if seqlen is None:
            seqlen = self.fm.block_size
        random.seed(self.seed + seqlen)
        return random.randint(100, 10000)
