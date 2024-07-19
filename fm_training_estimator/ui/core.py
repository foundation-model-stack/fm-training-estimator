# Local
from ..config import is_fsdp, parse
from ..memory import HybridEstimator
from ..throughput import MockSpeedEstimator
from ..utils import fmt_size


def run(config, lookup_data_path=None, model_path=None):

    res = {}
    fm, ta, ia = parse(config)

    est = HybridEstimator(fm, ta, ia, lookup_data_path, model_path)

    res["total_mem_estimate_og"] = est.get_total_mem_estimate()
    res["activation_memory_og"] = est.calculate_activation_memory()
    res["gradient_memory_og"] = est.calculate_gradient_memory()
    res["model_memory_og"] = est.calculate_model_memory()
    res["optimizer_memory_og"] = est.calculate_optimizer_memory()

    res["total_mem_estimate"] = fmt_size(res["total_mem_estimate_og"])
    res["activation_memory"] = fmt_size(res["activation_memory_og"])
    res["gradient_memory"] = fmt_size(res["gradient_memory_og"])
    res["model_memory"] = fmt_size(res["model_memory_og"])
    res["optimizer_memory"] = fmt_size(res["optimizer_memory_og"])

    if is_fsdp(ta):
        res["num_gpus"] = est.fsdp_est.get_number_of_gpus()

    speed_est = MockSpeedEstimator(fm, seed=10)
    res["tps"] = f"{speed_est.get_tps()} tokens/sec"

    return res
