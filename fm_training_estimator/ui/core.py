# Local
from ..config import is_fsdp, parse
from ..memory import HybridEstimator, HybridLoraEstimator, HybridQLoraEstimator
from ..throughput import HybridSpeedEstimator
from ..time import get_total_time
from ..tokens import TokenEstimator0, TokenEstimator2
from ..utils import fmt_size


def run(config, lookup_data_path=None, model_path=None):

    res = {}
    fm, ta, ia, da, la, qla = parse(config)

    token_est = None
    if da.te_approach == 0:
        token_est = TokenEstimator0(da)
    elif da.te_approach == 2:
        token_est = TokenEstimator2(da)

    if token_est != None:
        data_max_width = token_est.get_max_sample_length()
        if data_max_width < fm.block_size:
            fm.block_size = data_max_width

    if fm.technique == "lora":
        est = HybridLoraEstimator(fm, ta, ia, la, lookup_data_path, model_path)
    elif fm.technique == "qlora":
        est = HybridQLoraEstimator(fm, ta, ia, la, qla, lookup_data_path, model_path)
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
        elif fm.technique == "lora" or fm.technique == "qlora":
            res["num_gpus"] = est.num_gpus
        else:
            res["num_gpus"] = 1

        ia.numGpusPerPod = res["num_gpus"]

    # No suitable configuration found
    if res["num_gpus"] == -1:
        return {"error": "Input configuration is infeasible!"}

    speed_est = HybridSpeedEstimator(fm, ta, ia, lookup_data_path, model_path)
    res["tps"] = float(speed_est.get_tps())

    if token_est is not None:
        res["tokens_per_sample"] = int(
            token_est.get_estimated_batch_width(ta.per_device_train_batch_size)
        )
        res["total_tokens"] = int(token_est.get_total_tokens())

        # get the update tps for this estimate token width
        res["tps"] = float(speed_est.get_tps(res["tokens_per_sample"]))

        time_total, time_train = get_total_time(ta, ia, token_est, res["tps"], res["total_tokens"])
        res["time"] = time_total
        res["time_train"] = time_train

    return res
