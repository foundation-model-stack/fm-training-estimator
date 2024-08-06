# Standard
from dataclasses import dataclass, field

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
class PeftLoraConfig(LoraConfig):
    """dataclass for lora config

    Args:
        LoraConfig (_type_): directly imported from peft library
    """


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
