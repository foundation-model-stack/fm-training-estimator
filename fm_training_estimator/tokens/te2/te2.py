import json

# Third Party
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression

# Local
from ...config import DataArguments
from ...utils import logger
from ..te import TokenEstimator


def load_dataset_config_from_json(json_file_path):
    try:
        with open(json_file_path, "r") as file:
            config = json.load(file)
        print("Dataset configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


class TokenEstimator2(TokenEstimator):
    def __init__(self, da: DataArguments):
        if da.dataset_config_file is None:
            raise RuntimeError("Dataset configuration file has to be uploaded for TE2!")

        if da.dataset_config_file.endswith(".json"):
            logger.info("Parsing dataset configuration as local json file")
            contracts = load_dataset_config_from_json(da.dataset_config_file)
        else:
            raise RuntimeError("Please upload dataset configuration in correct JSON format!")

        # TODO: deal with formatted strings here
        contract = contracts[da.dataset_text_field]
        
        m = {}
        m[1] = contract["bs1"]
        batch_sizes = [2**i for i in range(1, 5) if 2**i <= contract["len"]]
        for bs in batch_sizes:
            m[bs] = contract[f"bs{bs}"]

        X = np.array([[i] for i in m.keys()])
        y = np.array(list(m.values()))

        self.contract = contract
        self.m = m
        self.reg = LinearRegression().fit(X, y)

    def get_total_tokens(self):
        return self.contract["total"]

    def get_estimated_batch_width(self, bs):
        if bs in self.m.keys():
            return self.m[bs]

        return self.reg.predict([[bs]])[0]

    def get_num_samples(self):
        return self.contract["len"]


# TODO: generate for all configs and splits
def GenerateTokenEstimator2Contract(dataset, config_name=None, split=None):
    if dataset.endswith(".json") or dataset.endswith(".jsonl"):
        logger.info("Parsing dataset as local json file")
        dataset = load_dataset("json", data_files={"train": dataset})["train"]
    else:
        dataset = load_dataset(dataset, name=config_name, split=split)

    print("Loading data in dataset...")

    feat_tokens = {}
    # TODO: run sampling instead of going through it all

    # mark all string features to generate contracts for
    for feat, f_val in dataset.features.items():
        if f_val.dtype == 'string':
            feat_tokens[feat] = []

    for item in tqdm(dataset):
        # loop over needed features
        for feat in feat_tokens.keys():
            feat_tokens[feat].append(int(len(item[feat]) / 3.6))

    contracts = {}
    for feat in feat_tokens.keys():
        tokens = np.sort(feat_tokens[feat])[::-1]

        contract = {}
        contract["len"] = len(tokens)
        contract["total"] = int(np.sum(tokens))
        contract["min"] = int(np.min(tokens))
        contract["max"] = int(np.max(tokens))
        contract["mean"] = round(np.mean(tokens), 2)
        contract["std"] = round(np.std(tokens), 2)

        contract["bs1"] = contract["mean"]

        # for bs = 2, 4, 6, 8, 16
        batch_sizes = [2**i for i in range(1, 5) if 2**i <= contract["len"]]
        for bs in batch_sizes:
            contract[f"bs{bs}"] = np.mean(tokens[:int(len(tokens)/bs)])

        contracts[feat] = contract

    return contracts
