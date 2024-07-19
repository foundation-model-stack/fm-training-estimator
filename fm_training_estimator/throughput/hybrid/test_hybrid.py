# Standard
from pathlib import Path

# Third Party
from pytest import raises

# Local
from ...config import parse
from ...regressor import XGBoostRegressor
from .hybrid import HybridSpeedEstimator

test_data2 = (Path(__file__).parent / "../../regressor/test_data/data2.csv").as_posix()


def test_hybrid_empty():
    fm, ta, ia = parse({})

    with raises(RuntimeError):
        _ = HybridSpeedEstimator(fm, ta, ia, None, None)


def test_hybrid_lookup():
    fm, ta, ia = parse(
        {
            "base_model_path": "ibm-granite/granite-7b-base",
            "per_device_train_batch_size": 4,
            "block_size": 512,
            "numGpusPerPod": 2,
        }
    )

    est = HybridSpeedEstimator(fm, ta, ia, test_data2, None)

    assert est.get_tps() == 500


def test_hybrid_reg(tmp_path):
    model_path = tmp_path / "test.model.json"
    reg = XGBoostRegressor()
    reg.train(test_data2, model_path, ["tokens_per_second", "memory", "memory_act"])

    fm, ta, ia = parse(
        {
            "base_model_path": "ibm-granite/granite-7b-base",
            "per_device_train_batch_size": 4,
            "block_size": 512,
            "numGpusPerPod": 4,
        }
    )

    est = HybridSpeedEstimator(fm, ta, ia, test_data2, model_path)

    assert est.get_tps() > 400
