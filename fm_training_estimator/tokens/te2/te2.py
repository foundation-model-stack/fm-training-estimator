# Third Party
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

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
                da.dataset, name=da.dataset_config_name, split=da.dataset_split
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
