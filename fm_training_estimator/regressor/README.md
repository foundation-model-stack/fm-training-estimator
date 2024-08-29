# regressor

This folder contains a few important pieces.

1. The format of data expected
2. Lookup module: to directly lookup an input configuration from existing dataset
3. XGBoost module: XGBoost based regressor

More regression modules can be added to this folder and then used in the various estimator modules. For example, the current modules are used in `memory/hybrid` and `throughput/hybrid`.

As more ML based modules are added here, the interfaces will be locked in. For now, the example to follow is the XGBoost module.

## Data formats

Data has to be in a common format for training regression modules and for runtime invocations. This is needed so that at runtime, we are able to correctly format the query to the lookup and regression modules.

Refer to the `data/` module for details on the data formats.
