# Standard
from pathlib import Path

# Local
from .xgboost import XGBoostRegressor

test_data1 = (Path(__file__).parent / "../test_data/data1.csv").as_posix()
test_data2 = (Path(__file__).parent / "../test_data/data2.csv").as_posix()


def test_reg_lifecycle(tmp_path):
    model_path = tmp_path / "test.model.json"

    reg = XGBoostRegressor()

    # train the model - and use it directly
    # includes saving to file
    reg.train(test_data1, model_path, ["tokens_per_second"])
    out = reg.run(["mercury-12b", "X100", 2, 4, 512])
    assert int(out[0]) < 1000

    # load the model from file
    reg1 = XGBoostRegressor()
    reg1.load(model_path)
    out1 = reg1.run(["mercury-12b", "X100", 2, 4, 512])
    assert int(out[0]) == int(out1[0])


def test_reg_multi(tmp_path):
    model_path = tmp_path / "test2.model.json"

    reg = XGBoostRegressor()

    # train the model - and use it directly
    # includes saving to file
    reg.train(test_data2, model_path, ["tokens_per_second", "memory", "memory_act"])
    out = reg.run(["ibm-granite/granite-7b-base", 2, 4, 512])

    assert len(out) == 1
    assert len(out[0]) == 3
