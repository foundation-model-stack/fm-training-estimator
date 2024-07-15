# ui

## cli

To use the cli:
```
python -m fm_training_estimator.ui.cli <path to config file>
```

## web

To use the web ui:
```
python -m fm_training_estimator.ui.web
```

to enable white listing of models, you can pass in the path of a txt file with one model per line. See the file `model_whitelist.txt` for an example. Use as:
```
python -m fm_training_estimator.ui.web ./model_whitelist.txt
```
