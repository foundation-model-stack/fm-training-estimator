# Standard
import math

# Local
from ...config import FMArguments, HFTrainingArguments
from ...utils import fmt_size
from ..full import FullParameterTuningEstimator


class FSDPEstimator:
    def __init__(
        self,
        fm_args: FMArguments,
        train_args: HFTrainingArguments,
        base: FullParameterTuningEstimator,
        gpuSize: int,
    ) -> None:
        self.base = base
        self.gpuSize = gpuSize
        self.num_of_model_params = self.base.num_of_model_params
        self.num_of_trainable_params = self.base.num_of_trainable_params
        self.optimizer = self.base.optimizer
        self.precision = self.base.precision
        self.s = self.base.s
        """fsdp options
            - `"full_shard"`: Shard parameters, gradients and optimizer states.
            - `"shard_grad_op"`: Shard optimizer states and gradients.
            - `"hybrid_shard"`: Apply `FULL_SHARD` within a node, and replicate parameters across nodes.
            - `"hybrid_shard_zero2"`: Apply `SHARD_GRAD_OP` within a node, and replicate parameters across nodes.
            - `"offload"`: Offload parameters and gradients to CPUs (only compatible with `"full_shard"` and
              `"shard_grad_op"`).
            - `"auto_wrap"`: Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.
        """
        self.fsdp_options = train_args.fsdp
        # ignores multi node training
        self.num_gpus = None

    def set_number_of_gpus(self, num_gpus):
        self.num_gpus = num_gpus

    def get_number_of_gpus(self):
        if self.num_gpus is None:
            self.estimate_number_of_gpus()

        return self.num_gpus

    def estimate_number_of_gpus(self):
        base_memory = (
            self.base.calculate_activation_memory(readable=False)
            + self.base.calculate_gradient_memory(readable=False)
            + self.base.calculate_optimizer_memory(readable=False)
        )
        if "shard_grad_op" in self.fsdp_options:
            return math.ceil(
                base_memory
                / (
                    self.gpuSize
                    - (
                        self.gpuSize * 0.01
                        + self.base.calculate_model_memory(readable=False)
                    )
                )
            )
        # leaving out 1% gap
        base_memory = (self.base.calculate_model_memory(readable=False)) + base_memory
        self.num_gpus = math.ceil(base_memory / (self.gpuSize - self.gpuSize * 0.01))
        return self.num_gpus

    def get_total_mem_estimate(self, readable: bool = False):
        size = (
            self.calculate_activation_memory()
            + self.calculate_gradient_memory()
            + self.calculate_model_memory()
            + self.calculate_optimizer_memory()
        )
        if readable:
            return fmt_size(size)
        return size

    def calculate_activation_memory(self, readable: bool = False):
        # activations are not sharded however, they are reduced by the minibatch size
        # minibatch is the per device batch size
        size = self.base.calculate_activation_memory(readable=False)
        if readable:
            return fmt_size(size)
        return size

    def calculate_gradient_memory(self, readable: bool = False):
        size = self.base.calculate_gradient_memory(readable=False) / (
            self.get_number_of_gpus()
        )
        if readable:
            return fmt_size(size)
        return size

    def calculate_optimizer_memory(self, readable: bool = False):
        size = self.base.calculate_optimizer_memory(readable=False) / (
            self.get_number_of_gpus()
        )
        if readable:
            return fmt_size(size)
        return size

    def calculate_model_memory(self, readable: bool = False):
        # at some point FSDP loads double the sharded model memory
        size = self.base.calculate_model_memory(readable=False)
        if not "shard_grad_op" in self.fsdp_options:
            size = size / (self.get_number_of_gpus())
        if readable:
            return fmt_size(size)
        return size
