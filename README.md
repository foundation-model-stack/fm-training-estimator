# FM Training Estimator

Estimators for Large Language Model Training.

Estimate resource consumption - memory, tokens, time etc for training and fine-tuning jobs using an hybrid of theory and learned regression models.

## Feature Matrix and Roadmap

| Technique          | Support            |
|--------------------|--------------------|
| Full (1 gpu)       | :heavy_check_mark: |
| FSDP (multi)       | :heavy_check_mark: |
| Lora (1 gpu)       | :heavy_check_mark: |
| QLora (1 gpu)      | Planned            |
| Speculators        | Planned            |
| Tensor Parallelism | Planned            |

### Time

Full learned approach. Coverage based on availability of training data.

### Memory

Hybrid theory + learned. Coverage of learned approach is subject to availability of training data.

### Tokens

Fully theory. Simulation based models available.

| Technique | Explanation                                    | Availability       |
|-----------|------------------------------------------------|--------------------|
| TE0       | Simulation based - slow but accurate           | :heavy_check_mark: |
| TE1       | Statistical                                    | Planned            |
| TE2       | Approximate - fast, light, reasonable accurate | Coming soon        |

## Usage

You can use the library `fm_training_estimator` as a Python package by installing it via pip, see [installation](#install), [build a regession model](#build-a-regression-model-for-learned-prediction-method) and [using the lirbary](#use-the-library-to-get-estimates). If you'd like to construct the estimator service with a [Web UI](#make-estimates-via-a-web-ui) via FastAPI or [build a docker image](#build-a-docker-container-image), clone the repository in your local machine before following the instructions in those sections.

Within your working directory, it is recommended to create a virtual environment to ensure no conflicts in dependencies.

```
python -m venv .venv
source .venv/bin/activate
```

### Install
```
pip install fm_training_estimator
```

### Build a regression model for learned prediction method

Now, prepare data in the expected format for lookup and regression. Some example data csv files are [here](https://github.com/foundation-model-stack/fm-training-estimator/tree/main/fm_training_estimator/regressor/test_data). Save your data file into `./workdir/data.csv`.

```
mkdir workdir
mv <data file> ./workdir/data.csv
```

Now, build a regression model using this data, using the provided make target:
```
from fm_training_estimator.regressor.xgboost.train import train 
train("./workdir/data.csv", "./workdir/model.json", ["tokens_per_second","memory","memory_act"])
```
This will create a model called `./workdir/model.json` which you can then use to estimate the resource consumption.

You can now run the estimator library, see below.

### Use the library to get estimates

For a full API reference, visit our [readthedocs](link).

Example code:
```python
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
print("With reg model: ", estimate_memory(est_input, model_path))


print("Estimating Time:....")
print("With only theory: ", estimate_time(est_input))
print("With reg model: ", estimate_time(est_input, model_path))

print("Estimating Tokens:....")
print("With only theory: ", estimate_tokens(est_input))
print("With reg model: ", estimate_tokens(est_input, model_path))
```

### Make estimates via a Web UI

To do this, first prepare a txt file called `model_whitelist.txt` in the `workdir/` with a list of model names, 1 per line. Note that these are the models on which you want to run the estimator to estimate their resource consumption. You can use the provided [example](https://github.com/foundation-model-stack/fm-training-estimator/blob/main/fm_training_estimator/ui/model_whitelist.txt) and place it in your `workdir`. Modify this list as needed.

Now, run the ui:
```
make run-web-ui
```
This will start the UI on `localhost:3000` port.

(The web ui has other options, not covered in this simple setup. If you want to skip the model whitelisting or change the port, directly run the UI as shown in the README in the `./fm_training_estimator/ui` folder.)

### Build a Docker Container Image

To build the estimator container image:

1. Make sure both `model.json` and `data.csv` files are present in the `workdir` folder.

2. Use this command to build and push the image:

```shell
make cbuild
make cpush # If you want to push to the container registry
```

3. Use this command to run the image:

```shell
docker run --rm -it -v "/path/to/input.json:/app/input.json" icr.io/ftplatform/fm_training_estimator:latest
```
