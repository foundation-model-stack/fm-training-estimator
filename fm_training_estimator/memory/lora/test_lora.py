# Standard

# Local
from ...config import parse
from ...utils import fmt_size
from .lora import LoraEstimator


def test_lora():
    fm, ta, _, _, la, _ = parse(
        {
            "base_model_path": "codellama/CodeLlama-13b-hf",
            "per_device_train_batch_size": 1,
            "torch_dtype": "bfloat16",
            "r": 8,
        }
    )
    est = LoraEstimator(fm, ta, la)

    assert est.calculate_optimizer_memory() < 100 * 1_000_000
