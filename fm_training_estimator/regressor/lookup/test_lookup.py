# Standard
from pathlib import Path

# Local
from .lookup import LookupRegressor

test_data1 = (Path(__file__).parent / "../test_data/data1.csv").as_posix()


def test_lookup():
    reg = LookupRegressor()

    reg.load(test_data1)

    res = reg.run(
        {
            "model_name": "mercury-12b",
            "gpu_model": "X100",
            "number_gpus": 2,
            "batch_size": 4,
            "seq_len": 512,
        }
    )

    assert res.shape[0] == 1
    assert res["tokens_per_second"][0] == 500

    # should return multiple entries
    res = reg.run(
        {
            "model_name": "mercury-12b",
            "gpu_model": "X100",
            "number_gpus": 2,
            "batch_size": 4,
        }
    )

    assert res.shape[0] == 3
