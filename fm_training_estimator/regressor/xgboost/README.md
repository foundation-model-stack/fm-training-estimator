# XGBoost Regressor

Train the regressor:
```
python -m fm_training_estimator.regressor.xgboost.train
```

Here is an example - run from the top level folder:
```
python -m fm_training_estimator.regressor.xgboost.train ./fm_training_estimator/regressor/test_data/data1.csv ./test.model.json ["train_tokens_per_second"]
```

This command will fail if the passed in y fields are not found in the input data.
