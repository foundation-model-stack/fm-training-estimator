---
title: Resource Estimator API
---

- **Author(s)**: Angel Luu (@aluu317)
- **Signer(s)**: Praveen Jayachandran, Ashok Pon Kumar Sree Prakash @ashokponkumar, Chander Govindarajan @ChanderG
- **Date (YYYY-MM-DD)**: 2024-10-01
- **Obsoletes ADRs**: N/A
- **Modified By ADRs**: N/A
- **Relevant Issues**: N/A

## Problem Context

Users of tuning/training stack currently have no way of estimate how much memory, time or cost it takes to run a training prior to training. They often hit OOM errors due to lack of memory. Users don't have enough information to make trade-off decisions on time vs. cost. Platform admins do not have any info to better schedule/pack jobs onto GPUs.

In order to be useful, the capability of estimating resources must be exposed to tuning/training users. The primary user of this service include training users and platform admins.

This ADR defines an API for a Resource Estimator service that provides an estimate of resource requirements for their training runs.

## Impact Table

| AI Functionality                                                                                            | Operational Functionality                                                                      |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Tuning Stack | APIs |

## Decision

- We will expose the API as REST using Open API, ADR [ref](https://github.ibm.com/ai-foundation/ai-foundation/blob/44d1163689b1aa1ca8ab6b9c571b73e6d05b9a0b/docs/current/adr/003-service-layer.md#decision).

- The REST API definitions will be hosted as Open Source at the repo [fm-training-estimator](https://github.com/foundation-model-stack/fm-training-estimator).

NOTE 1: We use REST API to mean an HTTP protocol server that uses standard HTTP verbs and supports Content-Type: application/json at a minimum. Full RESTful practices may be more strict

### REST API Alternatives
Allow kubernetes Custom Resource Definitions as API definitions. The Pros and Cons are discussed [here](https://github.ibm.com/ai-foundation/ai-foundation/blob/44d1163689b1aa1ca8ab6b9c571b73e6d05b9a0b/docs/current/adr/003-service-layer.md#rest-api-alternatives). 

It is noted that the Estimator service should support state (repeated calls for same base config, but slight tweaks). TODO: unsure?

## Consequences
-------- template ----------
Describe the resulting context, after applying the decision. All consequences should be listed here, not just the "positive" ones. A particular decision may have positive, negative, and neutral consequences, but all of them affect the team and project in the future. Be sure to include any impact on the platform's dependencies, technology choices, and Open Source community relationships.

Key things to include in this section:

- Impact on existing platform usage patterns, particularly any breaking changes
- Required changes in community relationships
- Expected changes in engineering work loads based on the decision (will this need a [research team and 5 years](https://xkcd.com/1425/)?)
- Changes to the supported ecosystems (introduction of new hardware, new runtime form-factor, etc...)
- Known risks of adopting this decision

-------- end template ----------


## High Level Design

- The REST API takes an input defined as `EstimateInput` data class (not all fields are required). This includes a list of instances of `Config` data class which in turnsincludes different types of configs (hf training args `HFArguments`, fms-hf-tuning additional args `FMArguments`, data args `DataArguments`, infrastructure args `InfraArguments` and peft lora args `PeftLoraConfig`), and `EstimatorConfig` with metadata parameters:

Example of an `EstimateInput` with all fields defined:
```json
{
  "estimator": { // EstimatorMetadata
    "base_data_path": "data.csv",
    "method": "theory", // theory, learned, hybrid
    "token_estimation_version": 0
  },
  "configs": [{ // list of [Config]
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

- The API exposes 4 endpoints: 

Endpoint `/api/memory` returns a `MemoryEstimate` as a JSON response:
```json
{
  "memory": { // MemoryEstimate
    "total_mem_estimate": "44.6 GiB",
    "activation_memory": "34.7 GiB",
    "gradient_memory": "2.5 GiB",
    "model_memory": "2.5 GiB",
    "optimizer_memory": "4.9 GiB",
    "num_gpus": 2
  }
}
```

Endpoint `/api/time` returns a `TimeEstimate` as a JSON response:
```json
{
  "time": { // TimeEstimate
    "time": "40s"
  }
}
```

Endpoint `/api/tokens` returns a `TokensEstimate` as a JSON response:
```json
{
  "tokens": { // TokensEstimate
    "tps": "5259.07373046875"
  }
}
```

Endpoint `/api/cost` returns a `CostEstimate` as a JSON response:
```json
{
  "cost": { // CostEstimate
    "usd": "" // todo: what is unit of cost? USD?
  }
}
```

Endpoint `/api/estimate` returns a `Estimate` that include all 4 types of estimates above as a JSON response:
```json
{
  "estimate": { // Estimate
    "memory_estimate": { // MemoryEstimate
      "total_mem_estimate": "44.6 GiB",
      "activation_memory": "34.7 GiB",
      "gradient_memory": "2.5 GiB",
      "model_memory": "2.5 GiB",
      "optimizer_memory": "4.9 GiB",
      "num_gpus": 2
    },
    "time": { // TimeEstimate
      "time": "40s"
    },
    "tokens": { // TokensEstimate
      "tps": "5259.07373046875"
    },
    "cost": { // CostEstimate
      "usd": "" // todo: what is unit of cost? USD?
    }
  }
}
```

- When more than 1 set of config is passed into the `EstimateInput`, the resulting estimate is an aggregated estimate of the job configs. TODO: unsure, is this supposed to mean total amount of time, memory, etc. Should it give a suggestion on order of jobs? How do we define job, job id? 