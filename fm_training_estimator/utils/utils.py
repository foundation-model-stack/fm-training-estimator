# Standard
from typing import Dict
import json
import math

# Third Party
from transformers import AutoConfig
from transformers.utils import logging
import yaml

logger = logging.get_logger("fm_training_estimator")


def unmarshal(path: str) -> Dict:
    """load data from the given filesystem path and return python dict

    Args:
        path (str): path to json or yaml file

    Returns:
        Dict: loaded data as python dictionary
    """
    if not path.endswith((".json", ".yaml", ".yml")):
        raise ValueError(
            "path to unmarshal should have either json or yaml extension, but got {path}".format(
                path=path
            )
        )
    with open(path, "r", encoding="utf8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def get_size_from_precision(precision: str) -> int:
    """return multiplier based on the precision

    Args:
        precision (str): paramter precision

    Returns:
        int: multiplier
    """
    if precision in ("float16", "bfloat16"):
        return 2
    if precision == "float32":
        return 4
    return 4


# https://stackoverflow.com/a/3155023
def get_human_readable_number(number: int) -> str:
    """return human readable format with denomination for the given number

    Args:
        number (int): number

    Returns:
        str: human readable string for number with denomination
    """
    denominations = ["", " Thousand", " Million", " Billion", " Trillion"]
    number = float(number)
    millidx = max(
        0,
        min(
            len(denominations) - 1,
            int(math.floor(0 if number == 0 else math.log10(abs(number)) / 3)),
        ),
    )

    return "{:.0f}{}".format(number / 10 ** (3 * millidx), denominations[millidx])


# https://stackoverflow.com/a/1094933
def fmt_size(size: int) -> str:
    """returns human readable format for the given size in bytes

    Args:
        size (int): number of bytes

    Returns:
        str: human readable string format for the bytes
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti"):
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}B"
        size /= 1024.0

    return f"{size:.1f} PiB"


def get_model_max_length(model_path: str) -> int:
    """return model's max sequence length by looking up its config

    Args:
        model_path (str): model path on filesystem or hugging face id

    Returns:
        int: max sequence length
    """
    config = AutoConfig.from_pretrained(model_path)
    n_positions = 4096
    if hasattr(config, "n_positions"):
        n_positions = config.n_positions
    elif hasattr(config, "max_position_embeddings"):
        n_positions = config.max_position_embeddings
    return n_positions
