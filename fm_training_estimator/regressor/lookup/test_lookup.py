# Standard
from pathlib import Path

# Local
from .lookup import LookupRegressor

test_data1 = (Path(__file__).parent / "../test_data/data1.csv").as_posix()
test_data2 = (Path(__file__).parent / "../test_data/data2.csv").as_posix()


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

    reg.load(test_data2)

    res = reg.run(
        {
            "model_name": "ibm-granite/granite-7b-base",
            "number_gpus": 2,
            "batch_size": 4,
            "seq_len": 512,
        }
    )

    assert res.shape[0] == 1
    assert res[0:1]["tokens_per_second"].item() == 500

    res = reg.run(
        {
            "model_name": "ibm-granite/granite-7b-base",
            "number_gpus": 2,
            "batch_size": 4,
            "seq_len": 1024,
        }
    )

    assert res.shape[0] == 1
    assert res[0:1]["tokens_per_second"].item() == 1000
