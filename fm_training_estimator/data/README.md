# Data

This module is used to standardize and version the supported data formats to be used both at train time (for the regression models) and at run time (the format to structure the data to feed to the lookup and the regression modules).

Since, we wish to support an ever evolving set of dataset features, the data format has been versioned into formats, such as "v1", "v2" and so on.

There are 3 integration points of this format:
1. The format of the data in the csv file for lookup. The names and order of columns, basically.
2. The feature names (with order) used to train any regression model to be used with the estimator.
3. The key values (with order) to be used at run time, to query one of the above 2 modules.

This module, locks in code, the exact expected format of data with version names. These names are mainly for human use, to refer to various formats. However, the job of this module is to automatically infer data format versions and adjust the data fields to make it easy for other modules to work with continuously changing data formats.

Specifcally:
1. For CSV files used in lookup, this module will check based on the header, the format version before using it.
2. Regression training is expected to use this module to bake the used data format into the model. This way, the model file can be safely shared and re-used. At model load, this format is extracted out and used in 3.
3. For runtime queries, this module provides helper functions to structure input data to fit the expected data format.

In the future, this module can also:
1. Provide validation functions to check any input data files/models.
2. Provide correction functions, to coerce input data files to the specified format.

## Formats

# v1: Name based data

For an example, look at `../regressor/test_data/data2.csv`. We need the following fields, in order:
```
model_name,number_gpus,batch_size,seq_len,tokens_per_second,memory,memory_act
```

Model_name is HF compatible name. All other fields are numbers.

Memory refers to total memory taken by that configuration in Bytes.
Memory_act refers to activation memory consumed by that configuration in Bytes.

# v2: Feature based data

For an example, look at `../regressor/test_data/data3.csv`. We need the following fields in order:
```
model_arch,model_hidden_size,model_intermediate_size,model_num_attn_heads,model_num_hidden_layers,model_num_key_value_heads,number_gpus,batch_size,seq_len,tokens_per_second,memory,memory_act
```

Notice how we no longer have the name of the model in the data. Instead, the first 6 fields refer to model configuration features which are now being used. All other fields are as in the `Name based data` format.
