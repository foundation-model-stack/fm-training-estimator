# Third Party
from fastapi import HTTPException

# First Party
from fm_training_estimator.config.arguments import (
    JobConfig,
    MemoryEstimateResponse,
    TimeEstimateResponse,
    TuningTechnique,
)

# Local
from ..config import is_fsdp, parse
from ..memory import HybridEstimator, HybridLoraEstimator
from ..throughput import HybridSpeedEstimator
from ..tokens import TokenEstimator0
from ..utils import fmt_size


def estimate_time(config: JobConfig, lookup_data_path=None, model_path=None):
    token_est = None
    if config.data.te_approach == 0:
        token_est = TokenEstimator0(config.data)

    speed_est = HybridSpeedEstimator(
        config.fm, config.hf_training, config.infra, lookup_data_path, model_path
    )
    # res["tps"] = float(speed_est.get_tps())

    time = ""
    if token_est is not None:
        tokens_per_sample = int(
            token_est.get_estimated_batch_width(
                config.hf_training.per_device_train_batch_size
            )
        )
        total_tokens = int(token_est.get_total_tokens())

        # get the update tps for this estimate token width
        tps = float(speed_est.get_tps(tokens_per_sample))

        time = total_tokens / tps

    if not time:
        raise HTTPException(
            status_code=501,
            detail="This te_approach is not implemented or has been disabled",
        )

    return TimeEstimateResponse(time)


def estimate_memory(config: JobConfig, lookup_data_path=None, model_path=None):
    if config.fm.technique == TuningTechnique.LORA:
        est = HybridLoraEstimator(
            config.fm,
            config.hf_training,
            config.infra,
            config.peft_lora,
            lookup_data_path,
            model_path,
        )
    else:
        est = HybridEstimator(
            config.fm, config.hf_training, config.infra, lookup_data_path, model_path
        )

    total_mem_estimate = fmt_size(float(est.get_total_mem_estimate()))
    activation_memory = fmt_size(float(est.calculate_activation_memory()))
    gradient_memory = fmt_size(float(est.calculate_gradient_memory()))
    model_memory = fmt_size(float(est.calculate_model_memory()))
    optimizer_memory = fmt_size(float(est.calculate_optimizer_memory()))
    num_gpus = config.infra.numGpusPerPod

    if num_gpus == 0:
        if config.fm.technique == TuningTechnique.FULL and is_fsdp(config.hf_training):
            num_gpus = est.fsdp_est.get_number_of_gpus()
        elif config.fm.technique == TuningTechnique.LORA:
            num_gpus = est.num_gpus
        else:
            num_gpus = 1

        config.infra.numGpusPerPod = num_gpus

    # No suitable configuration found
    if num_gpus == -1:
        raise HTTPException(
            status_code=422, detail="Input configuration is infeasible!"
        )

    return MemoryEstimateResponse(
        total_mem_estimate,
        activation_memory,
        gradient_memory,
        model_memory,
        optimizer_memory,
        num_gpus,
    )


def run(config, lookup_data_path=None, model_path=None):

    res = {}
    fm, ta, ia, da, la = parse(config)

    if config.fm.technique == "lora":
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
