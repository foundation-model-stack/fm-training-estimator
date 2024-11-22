# Standard
from pathlib import Path

# Local
from ...config import parse
from .te0 import TokenEstimator0

# trick to ensure running this with pytest works from root dir
test_data = (Path(__file__).parent / "te_test1.jsonl").as_posix()


def test_te_raw_hf_dataset():
    _, _, _, da, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "gpu_memory_in_gb": 80,
            "dataset": "super_glue",
            "dataset_config_name": "axb",
            "dataset_split": "test",
            "dataset_text_field": "Input###:\n {sentence1}",
        }
    )

    te = TokenEstimator0(da)

    assert te.get_num_samples() == 1104, "Number of samples in data should match"
    assert te.get_estimated_batch_width(4) < te.get_estimated_batch_width(
        8
    ), "Larger batches should have more padding"
    assert (
        abs(te.get_estimated_batch_width(1) - (te.get_total_tokens() / 1104)) < 1e6
    ), "Estimated batch width for BS=1 should be equal to mean of tokens"
    assert te.get_estimated_batch_width(1104) == max(
        te.tokens
    ), "For batch size equal to dataset length, estimated batch width should be equal to max length of tokens"


def test_te_raw_json():
    _, _, _, da, _, _ = parse(
        {
            "base_model_path": "ibm-granite/granite-8b-code-base",
            "gpu_memory_in_gb": 80,
            "dataset": test_data,
            "dataset_text_field": "{text}",
        }
    )

    te = TokenEstimator0(da)

    assert te.get_num_samples() == 9
    assert te.get_estimated_batch_width(1) < te.get_estimated_batch_width(2)
