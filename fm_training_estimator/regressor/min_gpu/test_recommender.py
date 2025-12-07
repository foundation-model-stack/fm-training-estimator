import pytest
from .recommender import MinGpuRecommenderCaller

MODEL_VERSION = "2.0.0"

valid_config = {
    "model_name": "llama3-8b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "batch_size": 16,
    "model_version": MODEL_VERSION,
}

valid_config_granite_3_3 = {
    "model_name": "granite-3.2-8b-instruct",
    "method": "full",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 1024,
    "batch_size": 4,
    "model_version": MODEL_VERSION,
}

infeasible_config = {
    "model_name": "llama3-8b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 32768,
    "batch_size": 128,
    "model_version": MODEL_VERSION
}

invalid_config = {
    "model_name": "granite-5.0-d453da",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "batch_size": 16,
    "model_version": MODEL_VERSION
}

model_not_trained= {
    "model_name": "granite-3.2-8b-instruct",
    "method": "full",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 4096.0,
    "batch_size": 128.0,
    "model_version": MODEL_VERSION
}

gpu_spelled_wrong = {
    "model_name": "llama-3.1-8b",
    "method": "full",
    "gpu_model": "NVIDIA A100-SXM4-80GB",
    "tokens_per_sample": 4096.0,
    "batch_size": 128.0,
    "model_version": MODEL_VERSION
}

recommender = None

result_with_valid_config = {
    "workers": 1,
    "gpus_per_worker": 4,
}

result_with_valid_config_granite_3_3= {
    "workers": 1,
    "gpus_per_worker": 1,
}

result_with_invalid_config = {
    "workers": -1,
    "gpus_per_worker": -1,
}

@pytest.fixture
def get_recommender_instance():
    return MinGpuRecommenderCaller()


def test_run_with_valid_config(get_recommender_instance):
    
    recommender = get_recommender_instance
    assert recommender.run(valid_config, "min_gpu") == result_with_valid_config

def test_run_with_valid_config_granite_3_3(get_recommender_instance):
    
    recommender = get_recommender_instance
    assert recommender.run(valid_config_granite_3_3, "min_gpu") == result_with_valid_config_granite_3_3

def test_run_with_invalid_config(get_recommender_instance):

    recommender = get_recommender_instance
    assert recommender.run(invalid_config, "min_gpu") == result_with_invalid_config

def test_run_with_infeasible_config(get_recommender_instance):

    assert get_recommender_instance.run(infeasible_config, "min_gpu") == result_with_invalid_config
