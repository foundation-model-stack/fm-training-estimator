# Standard

# Local
from ...config import parse
from .full import FullParameterTuningEstimator


def test_full():
    fm, ta, _ = parse({})
    est = FullParameterTuningEstimator(fm, ta)

    mm = est.calculate_model_memory()
    assert mm > 5 * 1_000_000_000
    assert mm < 15 * 1_000_000_000


def test_custom_model():
    fm, ta, _ = parse({"base_model_path": "ibm-granite/granite-8b-code-base"})
    est = FullParameterTuningEstimator(fm, ta)

    mm = est.calculate_model_memory()
    assert mm > 25 * 1_000_000_000
    assert mm < 35 * 1_000_000_000


def test_half_precision():
    fm, ta, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "torch_dtype": "float16",
        }
    )
    est = FullParameterTuningEstimator(fm, ta)

    mm = est.calculate_model_memory()
    assert mm > 10 * 1_000_000_000
    assert mm < 20 * 1_000_000_000
