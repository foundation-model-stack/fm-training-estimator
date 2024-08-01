# Local
from .model import extract_model_features


def test_extract_model_features():

    res = extract_model_features("ibm-granite/granite-3b-code-base")
    assert res["model_num_hidden_layers"] == 32

    res = extract_model_features("ibm-granite/granite-3b-code-base", fmt="list")
    assert res == ["LlamaForCausalLM", 2560, 10240, 32, 32, 32]

    res = extract_model_features("ibm-granite/granite-3b-code-base", fmt="csv")
    assert res == "LlamaForCausalLM,2560,10240,32,32,32"
