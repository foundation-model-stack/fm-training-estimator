# Standard
import os

# First Party
from fm_training_estimator.config.arguments import (
    EstimateInput,
    FMArguments,
    HFTrainingArguments,
    JobConfig,
)
from fm_training_estimator.sdk import estimate_memory

workdir_path = os.path.join(os.path.abspath(os.curdir), "workdir")

model_path = os.path.join(workdir_path, "model.json")
fm = FMArguments(
    base_model_path="ibm-granite/granite-7b-base",
    torch_dtype="float16",
    block_size=512,
)
hf_training = HFTrainingArguments(
    per_device_train_batch_size=4,
)
job_conf = JobConfig(
    hf_training,
    fm,
)
est_input = EstimateInput(job_configs=[job_conf])

print("With only theory: ", estimate_memory(est_input))
print("With reg model: ", estimate_memory(est_input, model_path))

hf_training.fsdp = "full_shard"

print("Using fsdp full shard")
print("With only theory: ", estimate_memory(est_input))
# print("With reg model: ", estimate_memory(est_input, model_path))
