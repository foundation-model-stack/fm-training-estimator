# Local
from ...config import parse
from .mock import MockSpeedEstimator


def test_mock_1():
    fm, _, _, _, _, _ = parse({"block_size": 512})
    est = MockSpeedEstimator(fm, seed=10)

    tps = est.get_tps()
    assert tps == 1355


def test_mock_2():
    fm, _, _, _, _, _ = parse({"block_size": 1024})
    est = MockSpeedEstimator(fm, seed=10)

    tps = est.get_tps()
    assert tps == 719
