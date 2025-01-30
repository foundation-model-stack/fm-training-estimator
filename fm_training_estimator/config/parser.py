# Standard
from typing import Dict, Tuple, Union

# Third Party
from transformers import HfArgumentParser

# Local
from ..utils import logger, unmarshal
from .arguments import (
    DataArguments,
    FMArguments,
    HFTrainingArguments,
    InfraArguments,
    PeftLoraConfig,
    PeftQLoraConfig,
)


def parse(
    config: Union[Dict, str]
) -> Tuple[
    FMArguments,
    HFTrainingArguments,
    InfraArguments,
    DataArguments,
    PeftLoraConfig,
    PeftQLoraConfig,
]:
    """parse config and return respective dataclass objects

    Args:
        config (Union[Dict, str]): path to config file or a config python dict

    Returns:
        Tuple[FMArguments, TrainingArguments, PeftLoraConfig, PeftQLoraConfig, PeftPromptTuningConfig]:
            dataclass objects
    """
    try:
        if not isinstance(config, (str, Dict)):
            raise TypeError(
                "provided config should be either path to a config file \
                    or a python:dict, but got {config_type}".format(
                    config_type=type(config)
                )
            )
        if isinstance(config, str):
            config = unmarshal(config)

        arg_parser = HfArgumentParser(
            [
                FMArguments,
                HFTrainingArguments,
                InfraArguments,
                DataArguments,
                PeftLoraConfig,
                PeftQLoraConfig,
            ]
        )

        return arg_parser.parse_dict(config)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "failed to parse the provided arguments from config {config}. error: {e}".format(
                config=config, e=e
            )
        )
