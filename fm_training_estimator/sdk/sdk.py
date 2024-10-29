# First Party
from fm_training_estimator.config.arguments import (
    CostEstimate,
    EstimateInput,
    JobConfig,
    MemoryEstimate,
    TimeEstimate,
    TokensEstimate,
)
from fm_training_estimator.memory.hybrid.hybrid import HybridEstimator
from fm_training_estimator.memory.lora.hybrid import HybridLoraEstimator
from fm_training_estimator.throughput.hybrid.hybrid import HybridSpeedEstimator
from fm_training_estimator.tokens.te0.te0 import TokenEstimator0

# Local
from ..config import is_fsdp
from ..utils import fmt_size


def _get_hybrid_estimator(conf: JobConfig, model_path: str = None):
    if conf.fm.technique == "lora":
        return HybridLoraEstimator(
            conf.fm,
            conf.hf_training,
            conf.infra,
            conf.peft_lora,
            None,
            model_path,
        )
    else:
        return HybridEstimator(conf.fm, conf.hf_training, conf.infra, None, model_path)


def estimate_memory(
    estimate_input: EstimateInput, model_path: str = None
) -> MemoryEstimate:
    """Estimate memory needed for training. This method uses hybdrid model by default.

    Args:
        estimate_input (fm_training_estimator.config.arguments.EstimateInput): the input for this estimation
            This input includes training job configs and optionally, metadata about this estimate run.
        model_path (str, optional): path to the trained xgboost model for the estimator to use for this run.

    Returns:
        fm_training_estimator.config.arguments.MemoryEstimate: the memory estimate of this run.

    """

    if estimate_input.job_configs is None or len(estimate_input.job_configs) == 0:
        raise ValueError("Did not receive a training job config")

    # Only going to process first job_config for now
    job_config = estimate_input.job_configs[0]

    est = _get_hybrid_estimator(job_config, model_path)

    total_mem_estimate = fmt_size(est.get_total_mem_estimate())
    activation_memory = fmt_size(est.calculate_activation_memory())
    gradient_memory = fmt_size(est.calculate_gradient_memory())
    model_memory = fmt_size(est.calculate_model_memory())
    optimizer_memory = fmt_size(est.calculate_optimizer_memory())

    num_gpus = job_config.infra.numGpusPerPod

    if num_gpus == 0:
        if job_config.fm.technique == "full" and is_fsdp(job_config.hf_training):
            num_gpus = est.fsdp_est.get_number_of_gpus()
        elif job_config.fm.technique == "lora":
            num_gpus = est.num_gpus
        else:
            num_gpus = 1

        job_config.infra.numGpusPerPod = num_gpus

    # No suitable configuration found
    if num_gpus == -1:
        raise ValueError("Input configuration is infeasible!")

    return MemoryEstimate(
        total_mem_estimate,
        activation_memory,
        gradient_memory,
        model_memory,
        optimizer_memory,
        num_gpus,
    )


def _estimate_tokens_and_time(
    conf: JobConfig,
    model_path: str = None,
    lookup_data_path: str = None,
) -> tuple[float, float]:
    token_est = None
    if conf.data.te_approach == 0:
        token_est = TokenEstimator0(conf.data)

    speed_est = HybridSpeedEstimator(
        conf.fm, conf.hf_training, conf.infra, lookup_data_path, model_path
    )

    estimated_tps = speed_est.get_tps()
    if estimated_tps is not None:
        tps = float(estimated_tps)
    else:
        # logger.warn("Could not calculate tps, defaulting to 1.")
        tps = 1

    if token_est is not None:
        tokens_per_sample = int(
            token_est.get_estimated_batch_width(
                conf.hf_training.per_device_train_batch_size
            )
        )
        total_tokens = int(token_est.get_total_tokens())

        # get the update tps for this estimate token width
        estimated_tps = speed_est.get_tps(tokens_per_sample)
        if estimated_tps is not None:
            tps = float(estimated_tps)
        else:
            # logger.warn("Could not calculate tps, defaulting to 1.")
            tps = 1

        time = total_tokens / tps
    else:
        # logger.warn("Could not get a total tokens to calculate time, setting time to 0.")
        time = 0
    return (tps, time)


def estimate_time(
    estimate_input: EstimateInput, model_path: str = None
) -> TimeEstimate:
    """Estimate time needed for training. This method uses hybdrid model by default.

    Args:
        estimate_input (fm_training_estimator.config.arguments.EstimateInput): the input for this estimation
            This input includes training job configs and optionally, metadata about this estimate run.
        model_path (str, optional): path to the trained xgboost model for the estimator to use for this run.

    Returns:
        fm_training_estimator.config.arguments.TimeEstimate: the time estimate of this run.

    """
    if estimate_input.job_configs is None or len(estimate_input.job_configs) == 0:
        raise ValueError("Did not receive a training job config")

    # Only going to process first job_config for now
    job_config = estimate_input.job_configs[0]

    _, time = _estimate_tokens_and_time(
        job_config, model_path, estimate_input.estimator_metadata.base_data_path
    )

    return TimeEstimate(time)


def estimate_tokens(
    estimate_input: EstimateInput, model_path: str = None
) -> TokensEstimate:
    """Estimate tokens throughput for a training. This method uses hybdrid model by default.

    Args:
        estimate_input (fm_training_estimator.config.arguments.EstimateInput): the input for this estimation
            This input includes training job configs and optionally, metadata about this estimate run.
        model_path (str, optional): path to the trained xgboost model for the estimator to use for this run.

    Returns:
        fm_training_estimator.config.arguments.TokensEstimate: the tokens throughput estimate of this run.

    """
    if estimate_input.job_configs is None or len(estimate_input.job_configs) == 0:
        raise ValueError("Did not receive a training job config")

    # Only going to process first job_config for now
    job_config = estimate_input.job_configs[0]

    tps, _ = _estimate_tokens_and_time(
        job_config, model_path, estimate_input.estimator_metadata.base_data_path
    )

    return TokensEstimate(tps)


def estimate_cost(
    estimate_input: EstimateInput, model_path: str = None
) -> CostEstimate:
    """Estimate cost for a training. This method uses hybdrid model by default. (Not yet supported)

    Args:
        estimate_input (fm_training_estimator.config.arguments.EstimateInput): the input for this estimation
            This input includes training job configs and optionally, metadata about this estimate run.
        model_path (str, optional): path to the trained xgboost model for the estimator to use for this run.

    Returns:
        fm_training_estimator.config.arguments.CostEstimate: the cost estimate of this run.

    """
    raise NotImplementedError("Not supported in this version.")
