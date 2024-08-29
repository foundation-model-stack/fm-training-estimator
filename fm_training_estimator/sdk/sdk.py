# Local
from ..config import is_fsdp, parse
from ..memory import HybridEstimator
from ..utils import fmt_size


def estimate_memory(config, reg_model=None):
    """
    If no reg model is passed in, falls back to pure theory approach.
    """
    res = {}

    # disable any token estimation
    config["te_approach"] = -1
    # set num gpus if needed
    if "numGpusPerPod" not in config:
        config["numGpusPerPod"] = 0

    fm, ta, ia, _ = parse(config)
    est = HybridEstimator(fm, ta, ia, None, reg_model)

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

    return res
