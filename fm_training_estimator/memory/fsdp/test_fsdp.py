# Local
from ...config import parse
from ..full import FullParameterTuningEstimator
from .fsdp import FSDPEstimator


def test_fsdp():
    fm, ta, ia, _, _, _ = parse(
        {"base_model_path": "ibm-granite/granite-8b-code-base", "gpu_memory_in_gb": 80}
    )

    base = FullParameterTuningEstimator(fm, ta)
    est = FSDPEstimator(fm, ta, base, 1024 * 1024 * 1024 * ia.gpu_memory_in_gb)

    est.set_number_of_gpus(1)
    mm1 = est.calculate_model_memory()

    est.set_number_of_gpus(2)
    mm2 = est.calculate_model_memory()

    assert mm1 == mm2 * 2

    est.set_number_of_gpus(4)
    mm4 = est.calculate_model_memory()

    assert mm1 == mm4 * 4
