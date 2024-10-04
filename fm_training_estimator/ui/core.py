# Local
from ..config import is_fsdp, parse
from ..memory import HybridEstimator, HybridLoraEstimator
from ..throughput import HybridSpeedEstimator
from ..tokens import TokenEstimator0
from ..utils import fmt_size


def run(config, lookup_data_path=None, model_path=None):

    res = {}
    fm, ta, ia, da, la = parse(config)

    if fm.technique == "lora":
        est = HybridLoraEstimator(fm, ta, ia, la, lookup_data_path, model_path)
    else:
        est = HybridEstimator(fm, ta, ia, lookup_data_path, model_path)

    res["total_mem_estimate_og"] = float(est.get_total_mem_estimate())
    res["activation_memory_og"] = float(est.calculate_activation_memory())
    res["gradient_memory_og"] = float(est.calculate_gradient_memory())
    res["model_memory_og"] = float(est.calculate_model_memory())
    res["optimizer_memory_og"] = float(est.calculate_optimizer_memory())

    res["total_mem_estimate"] = fmt_size(res["total_mem_estimate_og"])
    res["activation_memory"] = fmt_size(res["activation_memory_og"])
    res["gradient_memory"] = fmt_size(res["gradient_memory_og"])
    res["model_memory"] = fmt_size(res["model_memory_og"])
    res["optimizer_memory"] = fmt_size(res["optimizer_memory_og"])

    res["num_gpus"] = ia.numGpusPerPod

    if ia.numGpusPerPod == 0:
        if fm.technique == "full" and is_fsdp(ta):
            res["num_gpus"] = est.fsdp_est.get_number_of_gpus()
        elif fm.technique == "lora":
            res["num_gpus"] = est.num_gpus
        else:
            res["num_gpus"] = 1

        ia.numGpusPerPod = res["num_gpus"]

    # No suitable configuration found
    if res["num_gpus"] == -1:
        return {"error": "Input configuration is infeasible!"}

    token_est = None
    if da.te_approach == 0:
        token_est = TokenEstimator0(da)

    speed_est = HybridSpeedEstimator(fm, ta, ia, lookup_data_path, model_path)
    res["tps"] = float(speed_est.get_tps())

    if token_est is not None:
        res["tokens_per_sample"] = int(
            token_est.get_estimated_batch_width(ta.per_device_train_batch_size)
        )
        res["total_tokens"] = int(token_est.get_total_tokens())

        # get the update tps for this estimate token width
        res["tps"] = float(speed_est.get_tps(res["tokens_per_sample"]))

        res["time"] = res["total_tokens"] / res["tps"]

    return res
