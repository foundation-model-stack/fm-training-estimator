# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# Third Party
from dataclass_wizard import JSONWizard
from peft.tuners.lora import LoraConfig
from peft.tuners.prompt_tuning import PromptTuningConfig
from transformers import TrainingArguments


@dataclass
class PeftPromptTuningConfig(PromptTuningConfig):
    """dataclass for prompt tuning config

    Args:
        PromptTuningConfig (_type_): imported directly from peft library
    """


@dataclass
class PeftLoraConfig:
    """Dataclass for LoRA tuning config

    Not directly imported from peft LoraConfig due to complexity.
    """

    r: int = field(default=4, metadata={"help": ("Lora rank parameter")})

    lora_alpha: int = field(default=8)
    lora_dropout: float = field(default=0.1)
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class PeftQLoraConfig:
    """Dataclass for QLoRA tuning config"""

    quant_type: str = field(default="nf4")
    use_double_quant: bool = field(default=False)


@dataclass
class HFTrainingArguments(TrainingArguments):
    """HF trainer arguments

    Args:
        TrainingArguments (_type_): directly imported from transformers library
    """

    output_dir: str = field(
        default="./output", metadata={"help": ("temporary output dir for HF")}
    )


@dataclass
class InfraArguments:
    """dataclass for infrastructure arguments"""

    numGpusPerPod: int = field(
        default=0,
        metadata={
            "help": (
                "number of gpus requested per pod. Setting to 0 for auto-discover."
            )
        },
    )

    numPods: int = field(
        default=1,
        metadata={"help": ("number of pods requested")},
    )

    gpu_memory_in_gb: int = field(default=80, metadata={"help": ("GPU RAM in GBs")})

    gpuModel: str = field(
        default="A100",
        metadata={"help": ("model of gpu used")},
    )


@dataclass
class FMArguments:
    """dataclass to store additional args not covered by standard HF argument dataclasses"""

    base_model_path: str = field(
        default="ibm-granite/granite-3b-code-base",
        metadata={
            "help": (
                "Base Model location. Can be empty if output path has a checkpoint."
            )
        },
    )

    flash_attention_v2: bool = field(
        default=False,
        metadata={"help": ("It enable flash attention v2 for attention calculation.")},
    )

    lora_config: str = field(
        default=None, metadata={"help": ("LORA configuration json file path.")}
    )

    max_seq_length: int = field(
        default=2048,
        metadata={"help": ("model max sequence length.")},
    )

    block_size: int = field(
        default=2048,
        metadata={"help": ("Sequence length.")},
    )

    data_config_file: str = field(
        default="data_config.json",
        metadata={"help": ("Input files in glob format.")},
    )

    prompt_tuning_config: str = field(
        default=None, metadata={"help": ("Prompt tuning config json file path")}
    )

    torch_dtype: str = field(
        default="float32",
        metadata={
            "help": (
                "provide torch dtype for the model precision. \
                Choose one from float16, float32, bfloat16"
            )
        },
    )

    technique: str = field(
        default="full",
        metadata={"help": ("Fine-tuning technique being used")},
    )


@dataclass
class DataArguments:
    """dataclass to define args handling training data as input for estimation."""

    te_approach: int = field(
        default=0, metadata={"help": ("Approach to use for Token Estimation")}
    )

    dataset: str = field(
        default=None, metadata={"help": ("name of HF dataset or path to json file")}
    )

    dataset_text_field: str = field(
        default="text", metadata={"help": ("field of the dataset to use")}
    )

    dataset_split: str = field(
        default="test",
        metadata={"help": ("dataset split to use, in case of HF dataset")},
    )

    dataset_config_name: str = field(
        default=None,
        metadata={"help": ("dataset configuration to use, in case of HF dataset")},
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": ("allow dataset with a loading script")}
    )

    dataset_config_file: str = field(
        default=None,
        metadata={"help": ("dataset configuration file in case dataset is not available/provided")},
    )

class EstimatorMethod(Enum):
    """Enumerate different estimation models the FM Training Estimator is to use to make an estimation."""

    THEORY = "theory"
    """Theory model for estimation."""

    LEARNED = "learned"
    """Learned model for estimation, based on user provided training data."""

    HYBRID = "hybrid"
    """Hybrid model for estimation, a combination of theory and learned models."""


@dataclass
class EstimatorMetadata:
    """Metadata for the FM Training Estimator."""

    base_data_path: str = field(
        default=None, metadata={"help": ("path to the data path for training data")}
    )
    method: EstimatorMethod = field(
        default=EstimatorMethod.HYBRID,
        metadata={"help": ("enum method the estimator should use")},
    )
    token_estimation_version: str = field(
        default=0, metadata={"help": ("version of token estimator to use")}
    )


@dataclass
class JobConfig:
    """Dataclass that represents a set of different configs for a tuning job to make estimate on."""

    hf_training: HFTrainingArguments = field(default_factory=HFTrainingArguments)
    fm: FMArguments = field(default_factory=FMArguments)
    data: DataArguments = field(default_factory=DataArguments)
    infra: InfraArguments = field(default_factory=InfraArguments)
    peft_lora: PeftLoraConfig = field(default_factory=PeftLoraConfig)
    peft_qlora: PeftQLoraConfig = field(default_factory=PeftQLoraConfig)


@dataclass
class EstimateInput(JSONWizard):
    """
    The dataclass that is an input to a estimate function.
    It includes a list of different training job configs and metadata about the estimator.
    """

    job_configs: List[JobConfig]
    estimator_metadata: Optional[EstimatorMetadata] = None


@dataclass
class TimeEstimate:
    """The estimated time response to estimate_time function."""

    time: str
    train_time: str


@dataclass
class MemoryEstimate:
    """The estimated memory response to estimate_memory function."""

    total_mem_estimate: str
    activation_memory: str
    gradient_memory: str
    model_memory: str
    optimizer_memory: str
    num_gpus: int


@dataclass
class TokensEstimate:
    """The estimated token response to estimate_token function."""

    tps: float


@dataclass
class CostEstimate:
    """The estimated cost response to estimate_cost function."""

    usd: float


@dataclass
class Estimate:
    """The estimate response to estimate function, including time, memory, tokens and cost."""

    memory: MemoryEstimate
    time: TimeEstimate
    tokens: TokensEstimate
    cost: CostEstimate
