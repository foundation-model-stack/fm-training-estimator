# Time

Time taken for a training job consists for two main sub components:

1. Training time: actual time spent in the training process - forward pass, backward pass and so on.
2. Non-training time: other significant sources of times such as model load, model save etc.

## Training time

Training time is calculated in the estimator as a simple combination of two inputs:

1. `throughput`: that is number of tokens per second achieved by the training script.
2. `tokens`: the number of tokens to be processed for the given dataset under the given conditions.

Refer to the subcomponents in `../throughput` and `../tokens` for these calculations. Once we have both of these, a simple trivial division gives us the training time - albeit for a single epoch.

## Non-training time

This is made of the following components.

### Model load

In the beginning, a model must be loaded from disk, usually from files in Pytorch model formats or Hugging Face SafeTensor formats. This model may possibly fetched from the Hugging Face Hub, or available already on disk, cached or a from a local checkpoint format.

### Dataload time

Time take to load a dataset from files (typically json, jsonl or parquet) on disk.

### Checkpoint time

Every k steps or l epochs, a checkpoint may be saved to disk. There is a lot of research in this topic, to make this process faster.

