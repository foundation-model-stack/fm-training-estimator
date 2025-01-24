# Standard
from pathlib import Path

# Local
from ...config import parse
from .te2 import TokenEstimator2, GenerateTokenEstimator2Contract
from ..te0 import TokenEstimator0

# trick to ensure running this with pytest works from root dir
test_data = (Path(__file__).parent / "te_test1.jsonl").as_posix()
contract_axb = (Path(__file__).parent / "test_axb.contract.json").as_posix()
contract_test1 = (Path(__file__).parent / "test1.contract.json").as_posix()

def test_te_raw_hf_dataset():
    _, _, _, da, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "gpu_memory_in_gb": 80,
            "dataset": "super_glue",
            "dataset_config_name": "axb",
            "dataset_split": "test",
            "dataset_text_field": "{sentence1}",
            "dataset_config_file": contract_axb
        }
    )

    te = TokenEstimator2(da)
    te0 = TokenEstimator0(da)


    assert te.get_num_samples() == 1104, "Number of samples in data should match"
    assert te.get_estimated_batch_width(4) < te.get_estimated_batch_width(
        8
    ), "Larger batches should have more padding"
    assert (
        abs(te.get_estimated_batch_width(1) - (te.get_total_tokens() / 1104)) < 1e6
    ), "Estimated batch width for BS=1 should be equal to mean of tokens"
    assert (
        te0.get_estimated_batch_width(4) < te.get_estimated_batch_width(4)
    ), "TE0 must be less than TE2 for same batch size"


def test_te_raw_json():
    _, _, _, da, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "gpu_memory_in_gb": 80,
            "dataset": test_data,
            "dataset_text_field": "{text}",
            "dataset_config_file": contract_test1
        }
    )

    te = TokenEstimator2(da)
    te0 = TokenEstimator0(da)

    assert te.get_num_samples() == 9
    assert te.get_estimated_batch_width(1) < te.get_estimated_batch_width(2)
    assert te0.get_estimated_batch_width(2) < te.get_estimated_batch_width(2)

def test_te2_contract():
    _, _, _, da, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "gpu_memory_in_gb": 80,
            "dataset": test_data,
            "dataset_text_field": "{text}",
        }
    )

    contract = GenerateTokenEstimator2Contract(da.dataset)

    assert contract["text"]["len"] == 9
