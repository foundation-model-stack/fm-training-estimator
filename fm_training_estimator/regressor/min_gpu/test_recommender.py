from typing import Any, Generator
from fm_training_estimator.regressor.min_gpu.recommender import MinGpuRecommenderCaller
import pytest
from unittest.mock import Mock, patch, MagicMock
from .recommender import MinGpuRecommenderCaller

MODEL_VERSION = "3.1.0"

valid_config_min_gpu = {
    "model_name": "llama3-8b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "batch_size": 16,
    "model_version": MODEL_VERSION,
}

valid_config_avoid_oom = {
    "model_name": "llama3-8b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "per_device_train_batch_size": 1,
    "model_version": MODEL_VERSION,
}

valid_config_granite_3_3_min_gpu = {
    "model_name": "granite-3.2-8b-instruct",
    "method": "full",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 1024,
    "batch_size": 4,
    "model_version": MODEL_VERSION,
}

valid_config_granite_3_3_avoid_oom = {
    "model_name": "granite-3.2-8b-instruct",
    "method": "full",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 1024,
    "per_device_train_batch_size": 4,
    "model_version": MODEL_VERSION,
}

infeasible_config_min_gpu = {
    "model_name": "llama3-8b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 32768,
    "batch_size": 128,
    "model_version": MODEL_VERSION
}

infeasible_config_avoid_oom = {
    "model_name": "llama3-8b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 32768,
    "per_device_train_batch_size": 128,
    "model_version": MODEL_VERSION
}


invalid_config_min_gpu = {
    "model_name": "granite-3.2-8b-instruct",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "per_device_train_batch_size": 16,
    "model_version": MODEL_VERSION
}

invalid_config_avoid_oom = {
    "model_name": "granite-3.2-8b-instruct",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "batch_size": 16,
    "model_version": MODEL_VERSION
}

model_not_trained= {
    "model_name": "qwen-2.5B",
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

result_with_valid_config_avoid_oom = {
    "workers": 1,
    "gpus_per_worker": 1,
}

result_with_valid_config_granite_3_3= {
    "workers": 1,
    "gpus_per_worker": 1,
}

result_with_invalid_config = {
    "workers": -1,
    "gpus_per_worker": -1,
}

result_with_infeasible_config = {
    "workers": -1,
    "gpus_per_worker": -1,
}

@pytest.fixture
def get_recommender_instance_mock() -> Generator[MinGpuRecommenderCaller, Any, None]:
    """
    Returns a mock instance of MinGpuRecommenderCaller that mocks
    min_gpu_recommender calls (used for both min_gpu and avoid_oom modes).
    """
    with patch('fm_training_estimator.regressor.min_gpu.recommender.min_gpu_recommender') as mock_min_gpu, \
         patch('fm_training_estimator.regressor.min_gpu.recommender.ActuatorRegistry'):
        
        # Configure mock returns for successful recommendations
        mock_min_gpu.return_value = {
            "can_recommend": True,
            "workers": 1,
            "gpus_per_worker": 4
        }
        
        recommender = MinGpuRecommenderCaller()
        yield recommender

@pytest.fixture
def get_recommender_instance()-> Generator[MinGpuRecommenderCaller, Any, None]:
    """
     Returns a real instance of MinGpuRecommenderCaller that uses the real
     min_gpu_recommender function (used for both min_gpu and avoid_oom modes).
    """
    recommender = MinGpuRecommenderCaller()
    yield recommender 

# Test cases for min_gpu mode
def test_run_with_valid_config(get_recommender_instance):
    recommender = get_recommender_instance
    assert recommender.run(valid_config_min_gpu, "min_gpu") == result_with_valid_config

def test_run_with_valid_config_granite_3_3(get_recommender_instance):
    recommender = get_recommender_instance
    assert recommender.run(valid_config_granite_3_3_min_gpu, "min_gpu") == result_with_valid_config_granite_3_3

def test_run_with_invalid_config(get_recommender_instance):
    recommender = get_recommender_instance
    assert recommender.run(invalid_config_min_gpu, "min_gpu") == result_with_invalid_config

def test_run_with_infeasible_config(get_recommender_instance):
    assert get_recommender_instance.run(infeasible_config_min_gpu, "min_gpu") == result_with_invalid_config


# Test cases for avoid_oom mode
def test_run_with_valid_config_avoid_oom(get_recommender_instance):
    """Test avoid_oom mode with valid configuration"""
    recommender = get_recommender_instance
    assert recommender.run(valid_config_avoid_oom, "avoid_oom") == result_with_valid_config_avoid_oom

def test_run_with_valid_config_granite_3_3_avoid_oom(get_recommender_instance):
    """Test avoid_oom mode with valid granite 3.3 configuration"""
    recommender = get_recommender_instance
    assert recommender.run(valid_config_granite_3_3_avoid_oom, "avoid_oom") == result_with_valid_config_avoid_oom

def test_run_with_invalid_config_avoid_oom(get_recommender_instance):
    """Test avoid_oom mode with invalid configuration"""
    recommender = get_recommender_instance
    assert recommender.run(invalid_config_avoid_oom, "avoid_oom") == result_with_invalid_config

def test_run_with_infeasible_config_avoid_oom(get_recommender_instance):
    """Test avoid_oom mode with infeasible configuration"""
    assert get_recommender_instance.run(infeasible_config_avoid_oom, "avoid_oom") == result_with_infeasible_config


# Test case for min_gpu_recommender throwing exception
def test_min_gpu_recommender_exception():
    """Test that run() handles exceptions from min_gpu_recommender"""
    with patch('fm_training_estimator.regressor.min_gpu.recommender.min_gpu_recommender') as mock_min_gpu, \
         patch('fm_training_estimator.regressor.min_gpu.recommender.ActuatorRegistry'):
        
        mock_min_gpu.side_effect = Exception("min_gpu_recommender failed")
        
        recommender = MinGpuRecommenderCaller()
        result = recommender.run(valid_config_min_gpu, "min_gpu")
        
        # Should return -1 for both values when exception occurs
        assert result == result_with_invalid_config


# Test case for min_gpu_recommender throwing exception in avoid_oom mode
def test_avoid_oom_recommender_exception():
    """Test that run() handles exceptions from min_gpu_recommender in avoid_oom mode"""
    with patch('fm_training_estimator.regressor.min_gpu.recommender.min_gpu_recommender') as mock_min_gpu, \
         patch('fm_training_estimator.regressor.min_gpu.recommender.ActuatorRegistry'):
        
        mock_min_gpu.side_effect = Exception("min_gpu_recommender failed in avoid_oom mode")
        
        recommender = MinGpuRecommenderCaller()
        result = recommender.run(valid_config_min_gpu, "avoid_oom")
        
        # Should return -1 for both values when exception occurs
        assert result == result_with_invalid_config


# Test case for unknown recommendation type
def test_unknown_recommendation_type():
    """Test that run() raises ValueError for unknown recommendation type"""
    with patch('fm_training_estimator.regressor.min_gpu.recommender.ActuatorRegistry'):
        recommender = MinGpuRecommenderCaller()
        result = recommender.run(valid_config_min_gpu, "unknown_type")
        
        # Should return -1 for both values when ValueError is raised
        assert result == result_with_invalid_config


# Test case for recommender returning can_recommend=False
def test_cannot_recommend_min_gpu():
    """Test that run() handles can_recommend=False from min_gpu_recommender"""
    with patch('fm_training_estimator.regressor.min_gpu.recommender.min_gpu_recommender') as mock_min_gpu, \
         patch('fm_training_estimator.regressor.min_gpu.recommender.ActuatorRegistry'):
        
        mock_min_gpu.return_value = {"can_recommend": False}
        
        recommender = MinGpuRecommenderCaller()
        result = recommender.run(valid_config_min_gpu, "min_gpu")
        
        assert result == result_with_invalid_config


def test_cannot_recommend_avoid_oom():
    """Test that run() handles can_recommend=False from min_gpu_recommender in avoid_oom mode"""
    with patch('fm_training_estimator.regressor.min_gpu.recommender.min_gpu_recommender') as mock_min_gpu, \
         patch('fm_training_estimator.regressor.min_gpu.recommender.ActuatorRegistry'):
        
        mock_min_gpu.return_value = {"can_recommend": False}
        
        recommender = MinGpuRecommenderCaller()
        result = recommender.run(valid_config_min_gpu, "avoid_oom")
        
        assert result == result_with_invalid_config
