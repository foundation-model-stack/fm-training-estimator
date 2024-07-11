# Standard
from pathlib import Path

# Local
from .parser import parse

config_file_1 = (Path(__file__).parent / "./test_configs/config1.json").as_posix()


def test_parse_empty_dict():
    config = {}
    _, _, _ = parse(config)


def test_parse_dict():
    config = {"max_seq_length": 1023, "gpu_memory_in_gb": 40}
    fm, _, ia = parse(config)

    assert fm.max_seq_length == 1023
    assert ia.gpu_memory_in_gb == 40


def test_parse_file():
    fm, _, _ = parse(config_file_1)

    assert fm.max_seq_length == 1023
