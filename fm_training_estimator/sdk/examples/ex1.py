# Standard
import os

# First Party
from fm_training_estimator.config.arguments import (
    DataArguments,
    EstimateInput,
    EstimatorMetadata,
    FMArguments,
    HFTrainingArguments,
    InfraArguments,
    JobConfig,
)
from fm_training_estimator.sdk import (
    estimate_cost,
    estimate_memory,
    estimate_time,
    estimate_tokens,
)

workdir_path = os.path.join(os.path.abspath(os.curdir), "workdir")

model_path = os.path.join(workdir_path, "model.json")
lookup_data_path = os.path.join(workdir_path, "data.csv")

estimator_metadata = EstimatorMetadata(base_data_path=lookup_data_path)

fm = FMArguments(
    base_model_path="ibm-granite/granite-7b-base",
    torch_dtype="bfloat16",
    block_size=1024,
)
hf_training = HFTrainingArguments(
    per_device_train_batch_size=1, gradient_checkpointing=False
)
data = DataArguments(dataset="imdb", te_approach=0)
infra = InfraArguments(numGpusPerPod=1)
job_conf = JobConfig(hf_training, fm, data, infra)
est_input = EstimateInput(estimator_metadata=estimator_metadata, job_configs=[job_conf])

print("Estimating Memory:....")

print("With only theory: ", estimate_memory(est_input))
print("With reg model: ", estimate_memory(est_input, model_path))

hf_training.fsdp = "full_shard"

print("Using fsdp full shard")
print("With only theory: ", estimate_memory(est_input))
# print("With reg model: ", estimate_memory(est_input, model_path))


print("Estimating Time:....")
print("With only theory: ", estimate_time(est_input))
# print("With reg model: ", estimate_time(est_input, model_path))

print("Estimating Tokens:....")
print("With only theory: ", estimate_tokens(est_input))
# print("With reg model: ", estimate_tokens(est_input, model_path))
