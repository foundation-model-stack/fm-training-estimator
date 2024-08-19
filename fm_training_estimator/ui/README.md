# ui

## UI configuration options

### Lookup path

Path to file with raw CSV data. Look into the `regressor` folder to learn more about data formats.

### Model path

Path to model built using the `regressor` module.

### Use model features

Whether to use model name - just a string in the HF standard format, such as "ibm-granite/granite-7b-base" or model features.

Model features refers to metadata about the model, such as Architecure (eg, LlamaForCausalLM) and other configuration such as number of hidden layers, embedding dimension and so on. This approach extracts these features for an input model and uses that for the regression parts. This is the default.

Prefer the former if you have a small set of models of interest. The second approach is more generic and can handle new inputs better. 

## cli

To use the cli:
```
python -m fm_training_estimator.ui.cli <path to config file> -l <lookup file> -m <model file>
```
Lookup file and model file are optional and can be left out.

First train a memory model:
```
python -m fm_training_estimator.regressor.xgboost.train ./fm_training_estimator/regressor/test_data/data2.csv ./test.model.json '["tokens_per_second","memory","memory_act"]'
```

Run with all inputs:
```
python -m fm_training_estimator.ui.cli \
  ./fm_training_estimator/config/test_configs/config2.json \
  -l ./fm_training_estimator/regressor/test_data/data2.csv \
  -m ./test.model.json
```
`config2.json` is an example of the setup where Lookup would work. `config3.json` is an example where lookup will fail and the system will fall back to regression.

## api

Run the api:
```
make run-api
```

Now, you can get an estimate for the config using something like the following:
```
curl localhost:3000/api/estimate -d@<filename>
```
Notice that the request is a POST, since we need to pass in config json as a request body.

## web

To use the web ui:
```
python -m fm_training_estimator.ui.web
```

to enable white listing of models, you can pass in the path of a txt file with one model per line. See the file `model_whitelist.txt` for an example. Use as:
```
python -m fm_training_estimator.ui.web ./model_whitelist.txt
```

To enable lookup and regression based hybrid estimator:
```
python -m fm_training_estimator.ui.web ./model_whitelist.txt \
                                       ../regressor/test_data/data2.csv \
                                       ../../test.model.json
```

As with the cli version, first train the model to use.
