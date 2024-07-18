# ui

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
