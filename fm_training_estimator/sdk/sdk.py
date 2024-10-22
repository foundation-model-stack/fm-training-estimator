# First Party
from fm_training_estimator.config.arguments import EstimateInput, MemoryEstimate
from fm_training_estimator.memory.hybrid.hybrid import HybridEstimator
from fm_training_estimator.memory.lora.hybrid import HybridLoraEstimator

# Local
from ..config import is_fsdp, parse
from ..utils import fmt_size


def estimate_memory(
    estimate_input: EstimateInput, model_path: str = None
) -> MemoryEstimate:
    """
    Use hybdrid model by default.
    If no reg model is passed in (check in estimate_input.estimator_metadata), falls back to theory approach.
    """
    if estimate_input.job_configs is None or len(estimate_input.job_configs) == 0:
        raise ValueError("Did not receive a training job config")

    # Only going to process first job_config for now
    job_config = estimate_input.job_configs[0]

    # disable any token estimation
    job_config.data.te_approach = -1

    if job_config.fm.technique == "lora":
        est = HybridLoraEstimator(
            job_config.fm,
            job_config.hf_training,
            job_config.infra,
            job_config.peft_lora,
            None,
            model_path,
        )
    else:
        est = HybridEstimator(
            job_config.fm, job_config.hf_training, job_config.infra, None, model_path
        )

    total_mem_estimate = fmt_size(est.get_total_mem_estimate())
    activation_memory = fmt_size(est.calculate_activation_memory())
    gradient_memory = fmt_size(est.calculate_gradient_memory())
    model_memory = fmt_size(est.calculate_model_memory())
    optimizer_memory = fmt_size(est.calculate_optimizer_memory())

    if is_fsdp(job_config.hf_training):
        num_gpus = est.fsdp_est.get_number_of_gpus()
    else:
        num_gpus = 0
        # logger.info("Not enough info to estimate num_gpus, setting to 0.")

    return MemoryEstimate(
        total_mem_estimate,
        activation_memory,
        gradient_memory,
        model_memory,
        optimizer_memory,
        num_gpus,
    )
