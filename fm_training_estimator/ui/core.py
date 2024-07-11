# Local
from ..config import is_fsdp, parse
from ..memory import FSDPEstimator, FullParameterTuningEstimator
from ..utils import fmt_size


def run(config):

    res = {}
    fm, ta, ia = parse(config)

    est = FullParameterTuningEstimator(fm, ta)

    if is_fsdp(ta):
        est = FSDPEstimator(fm, ta, est, ia.gpu_memory_in_gb * 1024 * 1024 * 1024)
        if ia.numGpusPerPod != None:
            est.set_number_of_gpus(ia.numGpusPerPod)

    res["total_mem_estimate_og"] = est.get_total_mem_estimate(readable=False)
    res["activation_memory_og"] = est.calculate_activation_memory(readable=False)
    res["gradient_memory_og"] = est.calculate_gradient_memory(readable=False)
    res["model_memory_og"] = est.calculate_model_memory(readable=False)
    res["optimizer_memory_og"] = est.calculate_optimizer_memory(readable=False)

    res["total_mem_estimate"] = fmt_size(res["total_mem_estimate_og"])
    res["activation_memory"] = fmt_size(res["activation_memory_og"])
    res["gradient_memory"] = fmt_size(res["gradient_memory_og"])
    res["model_memory"] = fmt_size(res["model_memory_og"])
    res["optimizer_memory"] = fmt_size(res["optimizer_memory_og"])

    if is_fsdp(ta):
        res["num_gpus"] = est.get_number_of_gpus()

    return res
