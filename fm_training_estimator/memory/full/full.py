# Third Party
from transformers import AutoConfig, AutoTokenizer
from transformers.training_args import OptimizerNames

# Local
from ...config import FMArguments, HFTrainingArguments
from ...utils import fmt_size, get_size_from_precision, logger


class FullParameterTuningEstimator:
    def __init__(self, fm_args: FMArguments, train_args: HFTrainingArguments) -> None:
        # see https://huggingface.co/docs/transformers/v4.18.0/en/performance
        self.train_args = train_args
        self.fm_args = fm_args
        self.model_path = self.fm_args.base_model_path
        self.config = AutoConfig.from_pretrained(self.model_path)
        # check https://github.com/bigscience-workshop/bigscience/tree/6917a3b5fefcf439d3485ca184b4d9f6ab605150/math#model-sizing
        if hasattr(self.config, "n_embed"):
            self.h = self.config.n_embed
        elif hasattr(self.config, "n_embd"):
            self.h = self.config.n_embd
        elif hasattr(self.config, "hidden_size"):
            self.h = self.config.hidden_size
        h = self.h
        if hasattr(self.config, "n_layer"):
            l = self.config.n_layer
        elif hasattr(self.config, "num_hidden_layers"):
            l = self.config.num_hidden_layers
        self.l = l
        v = self.config.vocab_size
        self.v = v
        if hasattr(self.config, "n_head"):
            a = self.config.n_head
        elif hasattr(self.config, "num_attention_heads"):
            a = self.config.num_attention_heads
        self.a = a
        self.b = self.train_args.per_device_train_batch_size
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        n_positions = tokenizer.model_max_length
        if hasattr(self.config, "n_positions"):
            n_positions = self.config.n_positions
        if hasattr(self.config, "max_position_embeddings"):
            n_positions = self.config.max_position_embeddings
        self.model_max_length = n_positions
        self.s = min(self.fm_args.block_size, self.model_max_length)
        # trainable parameters in full paramter tuning
        self.num_of_model_params = l * (12 * h**2 + 13 * h) + v * h + 4 * h
        self.num_of_trainable_params = self.num_of_model_params

        # optimizers supported by transformer library
        self.optimizer = OptimizerNames(self.train_args.optim)
        self.precision = self._get_precision()

    def set_trainable_parameters(self, num_params):
        self.num_of_trainable_params = num_params

    def set_hidden_size(self, hidden_size):
        self.h = hidden_size

    def _get_precision(self) -> str:
        ## TODO: expand support for other precisions mentioned in TrainingArguments
        return self.fm_args.torch_dtype

    def calculate_activation_memory(self, readable: bool = False):
        # see https://blog.eleuther.ai/transformer-math/#activations-and-batch-size
        s = self.s
        b = self.b
        # dimension of the hidden representation
        a = self.a
        l = self.l
        h = self.h
        v = self.v
        # no tensor parallelism and sequence parallelism is considered at this point
        # activations stored in fp16 is assumed
        # (https://blog.eleuther.ai/transformer-math/#activations-and-batch-size)
        t = 1
        # TODO there are variations in  mem usage based on activation recomputation
        # we take the worst case scenario
        transformer_block_size = (s * b * h * l) * (
            10 + (24 / t) + (5 * (a * s) / (h * t))
        )
        # input embeddings + last norm + output layer
        # no pipeline parallelism
        v = self.config.vocab_size
        p = 1
        # peripheral_size = ((s*b*h*l) / t) * ((p / l) + ((p * 4 / l) * (1 + (v/h))))
        # print(fmt_size(peripheral_size))
        size = transformer_block_size

        if self.train_args.gradient_checkpointing:
            size /= self.l

        multiplier = 1
        if self.precision == "float32":
            logger.debug(f"Memory Full - Using multiplier 2 as precision is float32.")
            multiplier = 2
        elif self.precision == "float16" or self.precision == "bfloat16":
            logger.debug(
                f"Memory Full - Using multiplier 1 as precision is bfloat16 or float16."
            )
            multiplier = 1
        # print(s, b, h, l)
        # print(fmt_size(19 * s * b * h * l))
        size = size * multiplier
        # print(fmt_size(size / l))
        if readable:
            return fmt_size(size)
        return size

    def get_total_mem_estimate(self, readable: bool = False):
        # see https://blog.eleuther.ai/transformer-math/#distributed-training
        # TODO: fsdp is considered similar to Deepspeed zeros in terms of memory consumption
        # fsdp_sharding_strategy
        # FULL_SHARD (params, optim, and gradient) == deepspeed zero 3
        # SHARD_GRAD_OP (optim, and gradient) == deepspeed zero 2
        # NO_SHARD == DDP / deepspeed zero 0
        # HYBRID_SHARD (full shard in each node, like ddp across nodes) == deepspeed zero++ stage 3

        # however concrete formulation would be more helpful
        size = (
            self.calculate_activation_memory()
            + self.calculate_gradient_memory()
            + self.calculate_model_memory()
            + self.calculate_optimizer_memory()
        )
        if readable:
            return fmt_size(size)
        return size

    def calculate_gradient_memory(self, readable: bool = False):
        # see https://blog.eleuther.ai/transformer-math/#gradients
        multiplier = 0
        # TODO: gradient may not be in the same precision as the model
        # NOTE: there could be mixed precision as well
        # for mixed precision it is still fp32 computation
        if self.precision == "float32":
            multiplier = 4
        elif self.precision == "float16" or self.precision == "bfloat16":
            multiplier = 2
        else:
            raise ValueError("no support for the precision")
        size = self.num_of_trainable_params * multiplier
        if readable:
            return fmt_size(size)
        return size

    def calculate_model_memory(self, readable: bool = False):
        # TODO we did not consider mixed precision here
        # see https://huggingface.co/docs/transformers/v4.25.1/en/perf_train_gpu_one
        size = self.num_of_model_params * get_size_from_precision(self.precision)
        if readable:
            return fmt_size(size)
        return size

    def calculate_optimizer_memory(self, readable: bool = False):
        multiplier = 0
        # check https://github.com/huggingface/transformers/issues/22101
        ## check https://blog.eleuther.ai/transformer-math/#optimizer-states
        ## check https://huggingface.co/docs/transformers/v4.25.1/en/perf_train_gpu_one
        ## TODO: should detect 8-bit adamw if being used and compute
        if self.optimizer == OptimizerNames.ADAMW_TORCH or OptimizerNames.ADAMW_HF:
            # optimizer state is funciton of gradients/parameters dtype
            if self.precision == "float32":
                multiplier = 8
            elif self.precision == "float16" or self.precision == "bfloat16":
                multiplier = 4
        elif self.optimizer == OptimizerNames.SGD:
            multiplier = 4
        else:
            raise NotImplementedError("computation for optimizer is not implemented")
        size = self.num_of_trainable_params * multiplier
        if readable:
            return fmt_size(size)
        return size
