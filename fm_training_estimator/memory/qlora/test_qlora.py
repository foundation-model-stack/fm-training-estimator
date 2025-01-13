# Standard

# Local
from ...config import parse
from ...utils import fmt_size
from .qlora import QLoraEstimator

def test_qlora():
    fm, ta, _, _, la, qla = parse(
        {
            "base_model_path": "codellama/CodeLlama-13b-hf",
            "per_device_train_batch_size": 1,
            "torch_dtype": "bfloat16",
            "r": 8,
            "use_double_quant": False,
        }
    )
    est = QLoraEstimator(fm, ta, la, qla)

    assert est.calculate_optimizer_memory() < 100 * 1_000_000
