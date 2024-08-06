# Local
from .model import extract_model_features


def test_extract_model_features():

    # default dict format
    res = extract_model_features("ibm-granite/granite-3b-code-base")
    assert res["model_num_hidden_layers"] == 32

    res = extract_model_features("ibm-granite/granite-3b-code-base", fmt="list")
    assert res == ["LlamaForCausalLM", 2560, 10240, 32, 32, 32]

    # extract in csv format
    res = extract_model_features("ibm-granite/granite-3b-code-base", fmt="csv")
    assert res == "LlamaForCausalLM,2560,10240,32,32,32"

    # example from different format
    res = extract_model_features("ibm-granite/granite-20b-code-base", fmt="list")
    assert res == ["GPTBigCodeForCausalLM", 6144, 24576, 48, 52, 48]
