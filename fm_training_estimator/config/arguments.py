# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# Third Party
from peft.tuners.lora import LoraConfig
from peft.tuners.prompt_tuning import PromptTuningConfig
from transformers import TrainingArguments


@dataclass
class PeftPromptTuningConfig(PromptTuningConfig):
    """dataclass for promptuning config

    Args:
        PromptTuningConfig (_type_): imported directly from peft library
    """


@dataclass
class PeftLoraConfig:
    """Dataclass for lora config

    Not directly imported from peft LoraConfig due to complexity.
    """

    r: int = field(default=4, metadata={"help": ("Lora rank parameter")})

    lora_alpha: int = field(default=8)
    lora_dropout: float = field(default=0.1)
    target_modules: str = field(default="[q_proj, v_proj]")


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
        default=-1,
        metadata={"help": ("number of gpus requested per pod")},
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


class TuningTechnique(Enum):
    LORA = "lora"
    FULL = "full"


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

    technique: TuningTechnique = field(
        default=TuningTechnique.FULL,
        metadata={"help": ("Fine-tuning technique being used")},
    )


@dataclass
class DataArguments:
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


class EstimatorMethod(Enum):
    THEORY = "theory"
    LEARNED = "learned"
    HYBRID = "hybrid"


@dataclass
class EstimatorMetadata:
    base_data_path: str
    method: List[EstimatorMethod]
    token_estimation_version: str


@dataclass
class JobConfig:
    hf_training: HFTrainingArguments = field(default_factory=HFTrainingArguments)
    fm: FMArguments = field(default_factory=FMArguments)
    data: DataArguments = field(default_factory=DataArguments)
    infra: InfraArguments = field(default_factory=InfraArguments)
    peft_lora: PeftLoraConfig = field(default_factory=PeftLoraConfig)


@dataclass
class EstimateRequest:
    job_configs: List[JobConfig]
    estimator_metadata: Optional[EstimatorMetadata] = None


@dataclass
class TimeEstimateResponse:
    time: str


@dataclass
class MemoryEstimateResponse:
    total_mem_estimate: str
    activation_memory: str
    gradient_memory: str
    model_memory: str
    optimizer_memory: str
    num_gpus: int
