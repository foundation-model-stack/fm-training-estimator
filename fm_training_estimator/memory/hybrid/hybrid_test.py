# Standard
from pathlib import Path

# Local
from ...config import parse
from ...regressor import XGBoostRegressor
from .hybrid import HybridEstimator

test_data2 = (Path(__file__).parent / "../../regressor/test_data/data2.csv").as_posix()


def test_hybrid(tmp_path):

    model_path = tmp_path / "test.model.json"
    reg = XGBoostRegressor()
    reg.train(test_data2, model_path, ["tokens_per_second", "memory"])

    fm, ta, ia = parse(
        {
            "base_model_path": "ibm-granite/granite-7b-base",
            "gpu_memory_in_gb": 80,
            "fsdp": "full_shard",
            "per_device_train_batch_size": 4,
            "block_size": 512,
            "numGpusPerPod": 2,
        }
    )

    est = HybridEstimator(fm, ta, ia, test_data2, model_path)
    # Direct lookup example
    assert est.get_total_mem_estimate() == 20

    fm, ta, ia = parse(
        {
            "base_model_path": "ibm-granite/granite-7b-base",
            "gpu_memory_in_gb": 80,
            "fsdp": "full_shard",
            "per_device_train_batch_size": 4,
            "block_size": 512,
            "numGpusPerPod": 3,
        }
    )

    est = HybridEstimator(fm, ta, ia, test_data2, model_path)
    # Lookup fails - uses normal theory estimator
    assert est.get_total_mem_estimate() >= 50 * 1024 * 1024 * 1024
