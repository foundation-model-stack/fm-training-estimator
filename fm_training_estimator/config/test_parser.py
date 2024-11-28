# Standard
from pathlib import Path

# Local
from .parser import parse

config_file_1 = (Path(__file__).parent / "./test_configs/config4.json").as_posix()


def test_parse_empty_dict():
    config = {}
    _, _, _, _, _, _ = parse(config)


def test_parse_dict():
    config = {
        "max_seq_length": 1023,
        "gpu_memory_in_gb": 40,
        "block_size": 1023,
        "per_device_train_batch_size": 2,
        "dataset": "my-dataset",
    }
    fm, ta, ia, da, _, _ = parse(config)

    assert fm.max_seq_length == 1023
    assert ia.gpu_memory_in_gb == 40
    assert fm.block_size == 1023
    assert ta.per_device_train_batch_size == 2
    assert da.dataset == "my-dataset"


def test_parse_file():
    fm, ta, ia, da, _, _ = parse(config_file_1)

    assert da.dataset_config_file == "abc.json"
