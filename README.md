# FM Training Estimator

Estimators for Large Language Model Training.

Estimate resource consumption - memory, tokens, time etc for training and fine-tuning jobs using an hybrid of theory and learned regression models.

## Feature Matrix and Roadmap

| Technique          | Support            |
|--------------------|--------------------|
| Full (1 gpu)       | :heavy_check_mark: |
| FSDP (multi)       | :heavy_check_mark: |
| PT (1 gpu)         | Coming soon        |
| Lora (1 gpu)       | Coming soon        |
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

First, clone the repository in your local machine. Within the repository, create a virtual environment to ensure no conflicts in dependencies.

```
python -m venv .venv
```

### Install
```
make install
```

### Setup

Now, prepare data in the expected format for lookup and regression. Look at the csv files in `./fm_training_estimator/regressor/test_data/` for examples. Save this file into `./workdir/data.csv`.

```
mkdir workdir
mv <data file> ./workdir/data.csv
```

Now, build a regression model using this data, using the provided make target:
```
make build-model
```
This will create a model called `./workdir/model.json` which will be used to estimate the resource consumption.

You can now run the estimator, using one of the various UIs.


### Interacting

For a full list of UIs, look into the `./fm_training_estimator/ui` folder.

Easiest option is to run the Web UI.

To do this, first prepare a txt file called `model_whitelist.txt` in the `workdir/` with a list of model names, 1 per line. Note that these are the models on which you want to run the estimator to estimate their resource consumption. You can use the provided example listing using:
```
cp ./fm_training_estimator/ui/model_whitelist.txt ./workdir/
```
Modify this list as needed.

Now, run the ui:
```
make run-web-ui
```
This will start the UI on `localhost:3000` port.

(The web ui has other options, not covered in this simple setup. If you want to skip the model whitelisting or change the port, directly run the UI as shown in the README in the `./fm_training_estimator/ui` folder.)

### Container Image

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
