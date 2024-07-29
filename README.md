# FM Training Estimator
Estimators for Large Language Model Training

## Usage

### Setup

First install:
```
make install
```

Now, prepare data in the expected format for lookup and regression. Look at the csv files in `./fm_training_estimator/regressor/test_data/` for examples. Save this file into `./workdir/data.csv`.

```
mkdir workdir
mv <data file> ./workdir/data.csv
```

Now, build a regression model using this data, using the provided make target:
```
make build-model
```
This will create a model called `./workdir/model.json`.

You can now run the estimator, using one of the various UIs.

### Interacting

For a full list of UIs, look into the `./fm_training_estimator/ui` folder.

Easiest option is to run the Web UI.

To do this, first prepare a txt file called `model_whitelist.txt` in the `workdir/` with a list of model names, 1 per line. You can use the provided example listing using:
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
