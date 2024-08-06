# Standard
import logging

# Third Party
from transformers import AutoConfig


def get_model_max_length(model_path: str) -> int:
    """return model's max sequence length by looking up its config

    Args:
        model_path (str): model path on filesystem or hugging face id

    Returns:
        int: max sequence length
    """
    config = AutoConfig.from_pretrained(model_path)
    n_positions = 4096
    if hasattr(config, "n_positions"):
        n_positions = config.n_positions
    elif hasattr(config, "max_position_embeddings"):
        n_positions = config.max_position_embeddings
    return n_positions


def extract_model_features(model, fmt="dict"):
    """Given a model name in HF format, extract out the params of the model.

    Can return the data in one of supported formats.
    "dict": return a dictionary
    "list": return a list
    "csv": return a comma separated string of values
    """
    try:
        conf = AutoConfig.from_pretrained(model)
        conf = conf.to_dict()
        res = {}

        # TODO: include number of params here
        # need to refactor code out from Full Memory Estimator class

        res["model_arch"] = conf["architectures"][0]

        if "hidden_size" in conf:
            res["model_hidden_size"] = conf["hidden_size"]
        elif "n_embd" in conf:
            res["model_hidden_size"] = conf["n_embd"]
        elif "n_embed" in conf:
            res["model_hidden_size"] = conf["n_embed"]
        else:
            res["model_hidden_size"] = -1

        if "intermediate_size" in conf:
            res["model_intermediate_size"] = conf["intermediate_size"]
        elif "n_inner" in conf:
            res["model_intermediate_size"] = conf["n_inner"]
        else:
            res["model_intermediate_size"] = -1

        if "num_attention_heads" in conf:
            res["model_num_attn_heads"] = conf["num_attention_heads"]
        elif "n_head" in conf:
            res["model_num_attn_heads"] = conf["n_head"]
        else:
            res["model_num_attn_heads"] = -1

        if "num_hidden_layers" in conf:
            res["model_num_hidden_layers"] = conf["num_hidden_layers"]
        elif "n_layer" in conf:
            res["model_num_hidden_layers"] = conf["n_layer"]
        else:
            res["model_num_hidden_layers"] = -1

        if "num_key_value_heads" in conf:
            res["model_num_key_value_heads"] = conf["num_key_value_heads"]
        else:
            res["model_num_key_value_heads"] = res["model_num_attn_heads"]

    except Exception as e:
        logging.error(e)
        logging.warning("Returning empty response!")
        res = {}

    if fmt == "dict":
        return res

    if fmt == "list":
        return list(res.values())

    if fmt == "csv":
        out = ""
        for v in res.values():
            out += f"{v},"
        return out[:-1]

    logging.warning("Unknown format selected: ", fmt)
    return res
