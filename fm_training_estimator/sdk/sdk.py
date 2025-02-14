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
from fm_training_estimator.memory.qlora.hybrid import HybridQLoraEstimator
from fm_training_estimator.throughput.hybrid.hybrid import HybridSpeedEstimator
from fm_training_estimator.time import get_total_time
from fm_training_estimator.tokens.te0.te0 import TokenEstimator0

# Local
from ..config import is_fsdp
from ..utils import fmt_size, logger


def _get_hybrid_estimator(
    conf: JobConfig, model_path: str = None, lookup_data_path: str = None
):
    if conf.fm.technique == "lora":
        return HybridLoraEstimator(
            conf.fm,
            conf.hf_training,
            conf.infra,
            conf.peft_lora,
            lookup_data_path,
            model_path,
        )
    elif conf.fm.technique == "qlora":
        return HybridQLoraEstimator(
            conf.fm,
            conf.hf_training,
            conf.infra,
            conf.peft_lora,
            conf.peft_qlora,
            None,
            model_path,
        )
    else:
        return HybridEstimator(
            conf.fm, conf.hf_training, conf.infra, lookup_data_path, model_path
        )

def _update_seq_width(
    conf: JobConfig
) -> JobConfig:
    """
    Update the seq width based on the input dataset characteristics.

    This is only needed for memory and should not impact tps/tokens since those
    functions anyway operate on the input dataset.
    """

    token_est = None
    if conf.data.te_approach == 0:
        token_est = TokenEstimator0(conf.data)
    if conf.data.te_approach == 2:
        token_est = TokenEstimator2(conf.data)

    if token_est != None:
        data_max_width = token_est.get_max_sample_length()
        if data_max_width < conf.fm.block_size:
            conf.fm.block_size = data_max_width

    return conf

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

    # Update expected max width based on data
    job_config = _update_seq_width(job_config)

    if estimate_input.estimator_metadata:
        lookup_data_path = estimate_input.estimator_metadata.base_data_path
    if lookup_data_path is None:
        logger.warning(
            "SDK - No lookup data path given. Set it via estimator_metadata.base_data_path in input json. Proceeding with estimator with limited lookup ability."
        )

    est = _get_hybrid_estimator(job_config, model_path, lookup_data_path)

    total_mem_estimate = fmt_size(est.get_total_mem_estimate())
    activation_memory = fmt_size(est.calculate_activation_memory())
    gradient_memory = fmt_size(est.calculate_gradient_memory())
    model_memory = fmt_size(est.calculate_model_memory())
    optimizer_memory = fmt_size(est.calculate_optimizer_memory())

    num_gpus = job_config.infra.numGpusPerPod

    if num_gpus == 0:
        if job_config.fm.technique == "full" and is_fsdp(job_config.hf_training):
            num_gpus = est.fsdp_est.get_number_of_gpus()
        elif job_config.fm.technique == "lora" or job_config.fm.technique == "qlora":
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
    if conf.data.te_approach == 2:
        token_est = TokenEstimator2(conf.data)

    speed_est = HybridSpeedEstimator(
        conf.fm, conf.hf_training, conf.infra, lookup_data_path, model_path
    )

    estimated_tps = speed_est.get_tps()
    if estimated_tps is not None:
        tps = float(estimated_tps)
        logger.info("SDK - Initial estimated tps is %f", tps)
    else:
        logger.info("SDK - Could not calculate tps initially, defaulting to 1.")
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
            logger.info("SDK - Updated estimated tps after token width is %f", tps)
        else:
            logger.info(
                "SDK - Could not calculate tps after token width, defaulting to 1."
            )
            tps = 1

        # calculate full time here
        time = get_total_time(
            conf.hf_training, conf.infra, token_est, tps, total_tokens
        )
    else:
        time = (0, 0)
        logger.info(
            "SDK - Could not get a total tokens to calculate time, setting time to 0."
        )
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

    if estimate_input.estimator_metadata:
        lookup_data_path = estimate_input.estimator_metadata.base_data_path
    if lookup_data_path is None:
        logger.warning(
            "SDK - No lookup data path given. Set it via estimator_metadata.base_data_path in input json. Proceeding with estimator with limited lookup ability."
        )

    _, (time, train_time) = _estimate_tokens_and_time(
        job_config, model_path, estimate_input.estimator_metadata.base_data_path
    )

    return TimeEstimate(time, train_time)


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

    if estimate_input.estimator_metadata:
        lookup_data_path = estimate_input.estimator_metadata.base_data_path
    if lookup_data_path is None:
        logger.warning(
            "SDK - No lookup data path given. Set it via estimator_metadata.base_data_path in input json. Proceeding with estimator with limited lookup ability."
        )

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
