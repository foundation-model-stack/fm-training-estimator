---
title: Resource Estimator Library
---

- **Author(s)**: Angel Luu (@aluu317)
- **Signer(s)**: Praveen Jayachandran, Ashok Pon Kumar Sree Prakash @ashokponkumar, Chander Govindarajan @ChanderG
- **Date (YYYY-MM-DD)**: 2024-10-31
- **Obsoletes ADRs**: N/A
- **Modified By ADRs**: N/A
- **Relevant Issues**: N/A

## Problem Context

Users of tuning/training stack currently have no way of estimating how much memory, time or cost it takes to run a training. They often hit OOM errors due to lack of memory. Users don't have enough information to make trade-off decisions on time vs. cost. Platform admins do not have any info to better schedule/pack jobs onto GPUs.

In order to be useful, the capability of estimating resources must be exposed to tuning/training users. The primary user personas of this service include training users and platform admins.

This ADR defines a Resource Estimator Python Library that provides an estimate of resource requirements for training runs.

## Impact Table

| AI Functionality                                                                                            | Operational Functionality                                                                      |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Tuning Stack | APIs |

## Decision

- We will expose the resource estimator service as a Python library `fm_training_estimator`, hosted as Open Source at the repo [fm-training-estimator](https://github.com/foundation-model-stack/fm-training-estimator) and published to [PyPI](https://pypi.org/).
- This Python library can be installed and plugged into any UI backend or a docker image by a product team.
- The `fm_training_estimator` exposes 4 methods to calculate memory, time, tokens and cost. The method calls allows for user to pass training data as input for "learned" or "hybrid" model. If training data is missing, the "theory" is used.

### Alternatives to Python library deliverable
We have considered choices of:
- Alternative 1: A new docker image which has a FastAPI Server with a REST interface defined. When a product team integrates as a service, they can run this docker image, a server will run on localhost which can then be queried by GET/POST calls to do the estimates.

- Alternative 2: A new docker image with a python script similar to fms-hf-tuning, which accepts a JSON config and calls the necessary python scripts to get estimate and save results in a file.

Both alternatives provide more value to consumers. However does not provide the flexibility of how the library can be integrated and consumed.

## Consequences

- By using this library, users need to supply their own dataset for the estimator to generate a learned model, and assume the security and privacy of that data. They can use flight service plugin should that be applicable.
- The library can be used as backend component of a larger UI effort, or as part of a Docker image. The product teams can consume the library however they see fit and create their own build/update process.

## High Level Design

- The `EstimateInput` data class (not all fields are required) defines the set of configs the library will use to calculate the results. This includes a list of instances of `Config` data class which in turns includes different types of configs (hf training args `HFArguments`, fms-hf-tuning additional args `FMArguments`, data args `DataArguments`, infrastructure args `InfraArguments` and peft lora args `PeftLoraConfig`), and `EstimatorConfig` with metadata parameters. The input can be read from a json file using `--input_file_path` or `-f`.

Example of an `EstimateInput` with all fields defined:
```json
{
  "estimator": { // EstimatorMetadata
    "base_data_path": "data.csv",
    "method": "theory", // theory, learned, hybrid
    "token_estimation_version": 0
  },
  "job_configs": [{ // list of [JobConfig]
    "hf_training": { // HFArguments
      "output_dir": "./output"
    },
    "fm": { // FMArguments
      "base_model_path": "ibm-granite/granite-3b-code-base",
      "flash_attention_v2": "false",
      "lora_config": null,
      "max_seq_length": 2048,
      "block_size": 2048,
      "data_config_file": "data_config.json",
      "prompt_tuning_config": null,
      "torch_dtype": "float32",
      "technique": "full"
    },
    "data": { // DataArguments
      "te_approach": 0,
      "dataset": null,
      "dataset_text_field": "text",
      "dataset_split": "test",
      "dataset_config_name": null
    },
    "infra": { // InfraArguments
      "numGpusPerPod": 1,
      "numPods": 1,
      "gpu_memory_in_gb": 80,
      "gpuModel": "A100"
    },
    "peft_lora": { // PeftLoraConfig
      "r": 4,
      "lora_alpha": 8,
      "lora_dropout": 0.1,
      "target_modules": "[q_proj, v_proj]"
    }
  }]
}
```

- The API exposes 4 functions: 

Function `estimate_memory` returns a `MemoryEstimate`:
```python
{
  "memory": { # MemoryEstimate
    "total_mem_estimate": "44.6 GiB",
    "activation_memory": "34.7 GiB",
    "gradient_memory": "2.5 GiB",
    "model_memory": "2.5 GiB",
    "optimizer_memory": "4.9 GiB",
    "num_gpus": 2
  }
}
```

Function `estimate_time` returns a `TimeEstimate`:
```python
{
  "time": { # TimeEstimate
    "time": "40s"
  }
}
```

Function `estimate_tokens` returns a `TokensEstimate`:
```python
{
  "tokens": { # TokensEstimate
    "tps": "5259.07373046875"
  }
}
```

Function `estimate_cost` returns a `CostEstimate`:
```python
{
  "cost": { # CostEstimate
    "usd": "0.0"
  }
}
```

Function `estimate` returns a `Estimate` that include all 4 types of estimates above:
```python
{
  "estimate": { # Estimate
    "memory": { # MemoryEstimate
      "total_mem_estimate": "44.6 GiB",
      "activation_memory": "34.7 GiB",
      "gradient_memory": "2.5 GiB",
      "model_memory": "2.5 GiB",
      "optimizer_memory": "4.9 GiB",
      "num_gpus": 2
    },
    "time": { # TimeEstimate
      "time": "40s"
    },
    "tokens": { # TokensEstimate
      "tps": "5259.07373046875"
    },
    "cost": { # CostEstimate
      "usd": "0.0"
    }
  }
}
```