# Full Estimator

Estimating memory for a single GPU full fine-tuning.

## Experimental Features

### Gradient Checkpointing

How do we scale down Activation memory when Gradient Checkpointing is enabled? (The other 3 components are not impacted).

By examining Profiler output and looking at the code, we find that the checkpoint function (`torch.utils.checkpoint`) is called for each block, for eg, see: https://github.com/huggingface/transformers/blob/f5f1e52f6cf13cdf63ff25c311d33e2f2a842911/src/transformers/models/llama/modeling_llama.py#L984

This means that activations of a single block are stored when they are computed and once we are done with a block, just the inputs (which are stored in the `checkpoint` function) are retained to recompute activations for the backward pass.

So, a simple approximation is being used here - of scaling down the total activation memory to that consumed by a single layer or block in the Transformer arch.
