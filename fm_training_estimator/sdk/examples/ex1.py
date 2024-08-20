# First Party
from fm_training_estimator.sdk import estimate_memory

model_path = "../../../workdir/model.json"

config = {
    "base_model_path": "ibm-granite/granite-7b-base",
    "torch_dtype": "float16",
    "per_device_train_batch_size": 4,
    "block_size": 512,
}

print("With only theory: ", estimate_memory(config))
print("With reg model: ", estimate_memory(config, model_path))

config["fsdp"] = "full_shard"

print("With only theory: ", estimate_memory(config))
print("With reg model: ", estimate_memory(config, model_path))
