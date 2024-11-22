# Standard
from pathlib import Path

# Local
from ...config import parse
from ...regressor import XGBoostRegressor
from .hybrid import HybridEstimator

test_data2 = (Path(__file__).parent / "../../regressor/test_data/data2.csv").as_posix()
test_data3 = (Path(__file__).parent / "../../regressor/test_data/data3.csv").as_posix()


def test_hybrid(tmp_path):

    model_path = tmp_path / "test.model.json"
    reg = XGBoostRegressor()
    reg.train(test_data2, model_path, ["tokens_per_second", "memory", "memory_act"])

    fm, ta, ia, _, _, _ = parse(
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

    fm, ta, ia, _, _, _ = parse(
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
    # Lookup fails - uses Reg based approach
    assert est.get_total_mem_estimate() >= 10 * 1024 * 1024 * 1024

    grad_mem = est.calculate_gradient_memory()
    model_mem = est.calculate_model_memory()
    assert grad_mem >= 7 * 1024 * 1024 * 1024
    assert model_mem >= 7 * 1024 * 1024 * 1024
    assert grad_mem == model_mem


def test_use_model_features(tmp_path):

    model_path = tmp_path / "test.model.json"
    reg = XGBoostRegressor()
    reg.train(test_data3, model_path, ["tokens_per_second", "memory", "memory_act"])

    fm, ta, ia, _, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-7b-base",
            "gpu_memory_in_gb": 80,
            "fsdp": "full_shard",
            "per_device_train_batch_size": 16,
            "block_size": 1024,
            "numGpusPerPod": 4,
        }
    )

    est = HybridEstimator(fm, ta, ia, test_data3, None)

    # Direct lookup example should work as before
    assert est.get_total_mem_estimate() == 20

    fm, ta, ia, _, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "gpu_memory_in_gb": 80,
            "fsdp": "full_shard",
            "per_device_train_batch_size": 16,
            "block_size": 1024,
            "numGpusPerPod": 4,
        }
    )

    est = HybridEstimator(fm, ta, ia, test_data3, model_path)

    # Regression - based on model params
    # though we have only input model name here, we get predictions based on it's features
    assert est.calculate_activation_memory() < 30
