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


def parse_dataset_config(config):
    try:
        """Parse the JSON string into a Dataset dataclass."""
        contract = {}
        contract["len"] = config["number_of_tokens"]

        for field_data in config["fields"]:
            contract["min"] = contract.get("min", 0) + field_data["minimum_token_length"]
            contract["max"] = contract.get("max", 0) + field_data["maximum_token_length"]
            contract["std"] = contract.get("std", 0) + field_data["standard_deviation"] # Considering covariances between the fields to be 0
            contract["mean"] = contract.get("mean", 0) + field_data["mean_token_length"]
            contract["bs2"] = contract.get("bs2", 0) + field_data["50%ile_token_length"]
            contract["bs4"] = contract.get("bs4", 0) + field_data["75%ile_token_length"]
            contract["bs6"] = contract.get("bs6", 0) + field_data["83.33%ile_token_length"]
            contract["bs8"] = contract.get("bs8", 0) + field_data["87.5%ile_token_length"]
            contract["bs16"] = contract.get("bs16", 0) + field_data["93.75%ile_token_length"]
        contract["bs1"] = contract["mean"]
        contract["total"] = contract["mean"]*contract["len"]
        return contract

    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "failed to parse the dataset arguments from config {config}. error : \
                {e}".format(
                config=config, e=e
            )
        )
    

class TokenEstimator2(TokenEstimator):
    def __init__(self, da: DataArguments):
        if da.dataset_config_file is None:
            raise RuntimeError("Dataset configuration file has to be uploaded for TE2!")

        if da.dataset_config_file.endswith(".json"):
            logger.info("Parsing dataset configuration as local json file")
            dataset_config = load_dataset_config_from_json(da.dataset_config_file)
            contract = parse_dataset_config(dataset_config)
        else:
            raise RuntimeError("Please upload dataset configuration in correct JSON format!")
        
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
def GenerateTokenEstimator2Contract(dataset, config_name, split):
    if dataset.endswith(".json") or dataset.endswith(".jsonl"):
        logger.info("Parsing dataset as local json file")
        dataset = load_dataset("json", data_files={"train": dataset})["train"]
    else:
        dataset = load_dataset(dataset, name=config_name, split=split)

    tokens = []
    print("Loading data in dataset...")
    # TODO: run for all text fields here
    # TODO: run sampling instead of going through it all

    for item in tqdm(dataset):
        tokens.append(int(len(item[da.dataset_text_field]) / 3.6))

    tokens = np.sort(tokens)[::-1]

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

    return contract
