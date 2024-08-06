# regressor

This folder contains a few important pieces.

1. The format of data expected
2. Lookup module: to directly lookup an input configuration from existing dataset
3. XGBoost module: XGBoost based regressor

More regression modules can be added to this folder and then used in the various estimator modules. For example, the current modules are used in `memory/hybrid` and `throughput/hybrid`.

As more ML based modules are added here, the interfaces will be locked in. For now, the example to follow is the XGBoost module.

## Data formats

Data has to be in a standard format both for training regression modules and for runtime invocations. This section is the single source of truth for the data formats for the moment.

We have currently 2 data formats, documented below. It is important to ensure that the data format is the same between the raw files and trained models and the estimator `core`, which *will error out if data formats mismatch*.

# Name based data

For an example, look at `test_data/data2.csv`. We need the following fields, in order:
```
model_name,number_gpus,batch_size,seq_len,tokens_per_second,memory,memory_act
```

Model_name is HF compatible name. All other fields are numbers.

Memory refers to total memory taken by that configuration in Bytes.
Memory_act refers to activation memory consumed by that configuration in Bytes.

To use this data format, set `use_model_features` to `False` when running the `ui/core`.

# Feature based data

For an example, look at `test_data/data3.csv`. We need the following fields in order:
```
model_arch,model_hidden_size,model_intermediate_size,model_num_attn_heads,model_num_hidden_layers,model_num_key_value_heads,number_gpus,batch_size,seq_len,tokens_per_second,memory,memory_act
```

Notice how we no longer have the name of the model in the data. Instead, the first 6 fields refer to model configuration features which are now being used. All other fields are as in the `Name based data` format.

This is the current default. To use this data format, set `use_model_features` to `True` when running the `ui/core`.
