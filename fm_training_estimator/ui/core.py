# Local
from ..config import is_fsdp, parse
from ..memory import HybridEstimator
from ..throughput import HybridSpeedEstimator
from ..tokens import TokenEstimator0
from ..utils import fmt_size


def run(config, lookup_data_path=None, model_path=None, use_model_features=True):

    res = {}
    fm, ta, ia, da = parse(config)

    est = HybridEstimator(fm, ta, ia, lookup_data_path, model_path, use_model_features)

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

    token_est = None
    if da.te_approach == 0:
        token_est = TokenEstimator0(da)

    speed_est = HybridSpeedEstimator(
        fm, ta, ia, lookup_data_path, model_path, use_model_features
    )
    res["tps"] = speed_est.get_tps()

    if token_est is not None:
        res["tokens_per_sample"] = token_est.get_estimated_batch_width(
            ta.per_device_train_batch_size
        )
        res["total_tokens"] = token_est.get_total_tokens()

        # get the update tps for this estimate token width
        res["tps"] = speed_est.get_tps(res["tokens_per_sample"])

        res["time"] = res["total_tokens"] / res["tps"]

    return res
