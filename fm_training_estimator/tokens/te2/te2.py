# Third Party
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression

# Local
from ...config import DataArguments
from ...utils import logger
from ..te import TokenEstimator


class TokenEstimator2(TokenEstimator):
    def __init__(self, da: DataArguments):
        if da.dataset is None:
            raise RuntimeError("Dataset argument has to be filled in for TE2!")

        if da.dataset.endswith(".json") or da.dataset.endswith(".jsonl"):
            logger.info("Parsing dataset as local json file")
            dataset = load_dataset("json", data_files={"train": da.dataset})["train"]
        else:
            dataset = load_dataset(
                da.dataset,
                name=da.dataset_config_name,
                split=da.dataset_split,
                trust_remote_code=da.trust_remote_code
            )

        tokens = []
        print("Loading data in dataset...")
        for item in tqdm(dataset):
            tokens.append(int(len(item[da.dataset_text_field]) / 3.6))

        self.tokens = tokens

    def get_total_tokens(self):
        return np.sum(self.tokens)

    def get_estimated_batch_width(self, batch_size):
        tokens = np.sort(self.tokens)[::-1]
        return np.mean(tokens[:int(len(tokens)/batch_size)])

    def get_num_samples(self):
        return len(self.tokens)

def ExtractTokenEstimator2Contract(da: DataArguments):
    if da.dataset is None:
        raise RuntimeError("Dataset argument has to be filled in for TE2!")

    if da.dataset.endswith(".json") or da.dataset.endswith(".jsonl"):
        logger.info("Parsing dataset as local json file")
        dataset = load_dataset("json", data_files={"train": da.dataset})["train"]
    else:
        dataset = load_dataset(
            da.dataset, name=da.dataset_config_name, split=da.dataset_split
        )

    tokens = []
    print("Loading data in dataset...")
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

class TokenEstimator2FromContract:
    def __init__(self, contract: dict):
        self.contract = contract

        m = {}
        m[1] = contract["bs1"]
    
        batch_sizes = [2**i for i in range(1, 5) if 2**i <= contract["len"]]
        for bs in batch_sizes:
            m[bs] = contract[f"bs{bs}"]

        X = np.array([[i] for i in m.keys()])
        y = np.array(list(m.values()))

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