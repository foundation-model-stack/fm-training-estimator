# TE2

## Contract Generation

Examples:
```
python -m fm_training_estimator.tokens.te2.gen_contract --dataset imdb --output out1.contract.json
```
or
```
python -m fm_training_estimator.tokens.te2.gen_contract --dataset ./fm_training_estimator/tokens/te2/te_test1.jsonl --output out1.contract.json
```

This will output a single small contract file. This file should be later used with the estimator.
