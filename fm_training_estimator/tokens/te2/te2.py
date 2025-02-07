import json
import re

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

        baseline, fields = self.process_sample_format(da.dataset_text_field)

        self.baseline = baseline

        self.contract = {}
        self.m = {}
        self.reg = {}

        # For each field found in the input format, extract info from contract
        for field in fields:
            contract = contracts[field]
        
            m = {}
            m[1] = contract["bs1"]
            batch_sizes = [2**i for i in range(1, 5) if 2**i <= contract["len"]]
            for bs in batch_sizes:
                m[bs] = contract[f"bs{bs}"]

            X = np.array([[i] for i in m.keys()])
            y = np.array(list(m.values()))

            self.contract[field] = contract
            self.m[field] = m
            self.reg[field] = LinearRegression().fit(X, y)

    def process_sample_format(self, format_str):
        """
        Convert an input format string, into the constant baseline part and the fields used from the dataset.

        The baseline part is the number of tokens used in the static string part of the format.
        The fields are simply a list of matches of words in {}.

        For example, input format string maybe:
        'Below is a an instruction....
         ### Instruction
         {instruction}
         ### Input:
         {input}
         ### Response:'

        In the original data, we have contract information about "instruction" and "input" stored.
        In this function, we need to extract out how many tokens make the static portion and
        what fields are left over.
        """
        matches = re.findall(r'\{(.*?)\}', format_str)

        total = len(format_str)

        slot_len = 0
        for m in matches:
            # add 2 for the curly braces
            slot_len += 2 + len(matches)

        # number of tokens
        baseline = (total - slot_len) / 3.6

        return (total - slot_len, matches)

    def get_total_tokens(self):
        """
        Since each entry is also formatted with the fmt_string, we need to add the static portions.
        """
        total = 0
        num_samples = self.get_num_samples()

        # add all the common static tokens, one full set of baseline for each entey
        total += self.baseline * num_samples
        # now add the full set of tokens for fields that are present in here
        for con in self.contract.values():
            total += con["total"]

        return total

    def get_estimated_batch_width(self, bs):
        """
        Since multiple fields make up a single entry, we predict average size of each and 
        also add the baseline width to it.
        """
        width = self.baseline

        for field in self.contract.keys():
            m = self.m[field]
            reg = self.reg[field]
            if bs in m.keys():
                width += m[bs]
            else:
                width += reg.predict([[bs]])[0]

        return width

    def get_num_samples(self):
        # length of all contracts will be same
        con = list(self.contract.values())[0]
        return con["len"]

    def get_max_sample_length(self):
        res = self.baseline
        # this is a very pessimistic view, and not the actual worst case
        for con in self.contract.values():
            res += con["max"]

        return res


# TODO: generate for all configs and splits
def GenerateTokenEstimator2Contract(dataset, config_name=None, split=None, sample_percent=None):
    if dataset.endswith(".json") or dataset.endswith(".jsonl"):
        logger.info("Parsing dataset as local json file")
        dataset = load_dataset("json", data_files={"train": dataset})["train"]
    else:
        dataset = load_dataset(dataset, name=config_name, split=split)

    print("Loading data in dataset...")

    feat_tokens = {}
    # TODO: run sampling instead of going through it all
    num_items = len(dataset)
    if sample_percent != None:
        if sample_percent > 0 and sample_percent <= 100:
            num_items = int(num_items * sample_percent/100)

    # mark all string features to generate contracts for
    for feat, f_val in dataset.features.items():
        if f_val.dtype == 'string':
            feat_tokens[feat] = []

    seen_items = 0
    for item in tqdm(dataset):
        # loop over needed features
        for feat in feat_tokens.keys():
            feat_tokens[feat].append(int(len(item[feat]) / 3.6))

        seen_items += 1
        if seen_items >= num_items:
            break

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

        # if we are in sampling mode, rescale the stats
        if num_items != len(dataset):
            contract["len"] = len(dataset)
            contract["total"] = int(contract["total"]*len(dataset)/num_items)

    return contracts
